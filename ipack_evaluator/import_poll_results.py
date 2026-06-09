import pandas as pd
from pathlib import Path
from c_evaluator.utils import get_cache_dir
import json
from itertools import permutations
import numpy as np


class PollEvaluator:
    def __init__(self, csv_answer_file: Path, hope_mapping_path: Path = None):
        # open the file
        self.data = pd.read_csv(csv_answer_file)
        self.data.rename(
            columns={self.data.columns[1]: "answers"}, inplace=True
        )
        self.hope_mapping_path = hope_mapping_path
        if self.hope_mapping_path:
            self.hope_mapping = json.load(open(hope_mapping_path))

        # get the second column
        answers = self.data.loc[:, "answers"].dropna()
        self.gt_matrix = None
        self.gt_matrix, _ = self.get_comparison_matrix(answers)

        # fix trace to 0.5
        for idx in self.gt_matrix.index:
            self.gt_matrix.loc[idx, idx] = 0.5

        self.hash_table = {}

    def get_comparison_matrix(
        self, answers: pd.DataFrame
    ) -> tuple[pd.DataFrame, int]:
        """
        Answers is a list of strings, each string is a list of grocery items.
        The answers are separated by ",".
        We wannt to compile a matrix, which accumulates the number of times,
        a grocery is above another grocery.
        Example: If bananas apear before bottes, we add 1 to the matrix.
        """

        if self.gt_matrix is None:
            answers_list = list(answers.loc[0].split(","))
        else:
            answers_list = list(self.gt_matrix.index)
        answers_list.sort()
        number_of_votes = len(answers)

        # save the answers
        self.write_answers(answers_list)

        # create a matrix
        matrix = pd.DataFrame(data=0, index=answers_list, columns=answers_list)

        subset_of_groceries_utilized_in_answers = set()
        # iterate over the answers
        for answer in answers:
            groceries_sequence = answer.split(",")
            for i, grocery in enumerate(groceries_sequence):
                if grocery not in matrix.index:
                    print(f"Warning!! Grocery {grocery} not in gt matrix!")
                    continue
                subset_of_groceries_utilized_in_answers.add(grocery)
                for other_grocery in groceries_sequence[i + 1 :]:
                    if other_grocery not in matrix.index:
                        print(
                            f"Warning!! Grocery {other_grocery} not in gt matrix!"
                        )
                        continue
                    matrix.loc[
                        grocery, other_grocery
                    ] += 1  # grocery is above other_grocery

        # divide by the number of votes
        matrix = matrix / number_of_votes

        return matrix, subset_of_groceries_utilized_in_answers

    def get_max_score(self, matrix_in: pd.DataFrame) -> float:
        # iterate over rows and columns
        max_score = 0

        for row in matrix_in.index:
            for column in matrix_in.columns:
                if (
                    self.gt_matrix.loc[row, column]
                    > self.gt_matrix.loc[column, row]
                ):
                    max_score += matrix_in.loc[row, column]
                else:
                    max_score += matrix_in.loc[column, row]

        # we iterate over every cell twice, so we need to divide by 2
        max_score /= 2

        return max_score

    def get_max_score_subset(self, sequence) -> float:
        groceries = sequence["answers"][0].split(",")

        # get hash for groceries
        groceries_hash = hash(sequence["answers"][0])

        # check if we have already calculated the max score for this sequence
        if groceries_hash in self.hash_table:
            print("Retrieving hash!")
            return self.hash_table[groceries_hash]

        # delet all groceries that are not in the gt matrix
        groceries = [
            grocery for grocery in groceries if grocery in self.gt_matrix.index
        ]

        sub_mat = self.gt_matrix.loc[groceries, groceries]
        perf_seq = self.matrix_to_sequence(sub_mat)

        num_perf = len(perf_seq["answers"][0].split(","))
        num_groceries = len(groceries)

        if num_perf != num_groceries:
            # find which groceries occure how many times
            perf_seq = perf_seq["answers"][0].split(",")
            groceries_count = {}
            for grocery in groceries:
                if grocery not in groceries_count:
                    groceries_count[grocery] = 0
                groceries_count[grocery] += 1

            # delete all groceries with count 1
            groceries_count_new = {}
            for grocery, count in groceries_count.items():
                if count == 1:
                    continue
                groceries_count_new[grocery] = count
            groceries_count = groceries_count_new

            # all others: add them after their first occurence
            perf_s_copy = perf_seq.copy()
            for grocery, count in groceries_count.items():
                if count > 1:
                    idx = perf_seq.index(grocery)
                    for i in range(count - 1):
                        perf_seq.insert(idx + 1, grocery)

            if len(perf_seq) != num_groceries:
                raise ValueError(
                    f"Length of sequence is {len(perf_seq)} but should be {num_groceries}"
                )

            perf_seq = pd.DataFrame({"answers": [",".join(perf_seq)]})

        max_score, _, _ = self.get_score(perf_seq)[0]

        # save the max score for this sequence
        self.hash_table[groceries_hash] = max_score
        print("Found new hash!")

        return max_score

    def get_score(
        self, sequence: pd.DataFrame, is_matrix: bool = False
    ) -> float:
        """
        Get the similarity score of the sequence compared to the human answers.

        Args:
            sequence (pd.DataFrame): A DataFrame with the list of ansers in a
                comma seperated manner. The first one is the lowest one (thus
                the first one to be loaded). WARNING: This is inverse to the
                human answers.
        """
        if not is_matrix:
            # invert the list to match internal standard (first is highest)
            sequence = (
                sequence.loc[:, "answers"]
                .str.split(",")
                .apply(lambda x: x[::-1])
            )
            sequence = sequence.apply(lambda x: ",".join(x))

            # get the sequence
            # sequence = sequence.loc[:, "answers"]

            # get the matrix of the sequence
            # raise NotImplementedError(
            #    "Changed this fast. doublecheck before using again!"
            # )

            sequence_matrix, subset_of_groceries_utilized_in_answer = (
                self.get_comparison_matrix(sequence)
            )
        else:
            # get zero matrix of the size of the gt matrix
            sequence_matrix = pd.DataFrame(
                data=0,
                index=self.gt_matrix.index,
                columns=self.gt_matrix.columns,
            )

            # copy the values from the input matrix to the sequence matrix
            for row in sequence.index:
                for col in sequence.columns:
                    if (
                        row in sequence_matrix.index
                        and col in sequence_matrix.columns
                    ):
                        sequence_matrix.loc[row, col] = sequence.loc[row, col]

            subset_of_groceries_utilized_in_answer = set(sequence.index)

        # element wise multiply the matrices and sum all elements
        # score = (sequence_matrix * self.gt_matrix).sum().sum()

        # element wise exponentiation: gt_matrix^sequence_matrix
        score = self.gt_matrix ** sequence_matrix

        # make > 0.4 to 1 and <= 0.6 to 0 -> how many constraints do we violate?
        gt_matrix_abs = self.gt_matrix.map(lambda x: 1 if x > 0.4 else 0)

        # how many times do we violate the constraints?
        n_correct = (gt_matrix_abs * sequence_matrix).sum().sum()
        n_total = sequence_matrix.sum().sum()

        n_violations = n_total - n_correct

        # replace all 0 with 1
        score = score.replace(0, 1)

        # multiply all elements
        score = score.prod().prod()

        return (
            score,
            subset_of_groceries_utilized_in_answer,
            n_violations,
            n_total,
        )

    def matrix_to_sequence(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """Convert matrix to sequence using topological sorting.

        Args:
            matrix (pd.DataFrame): _description_

        Returns:
            pd.DataFrame: _description_
        """

        # create a copy of the matrix
        matrix = matrix.copy()

        # create a list to store the sequence
        sequence = []

        # iterate over the matrix until it is empty
        while not matrix.empty:
            # get the sum of all columns
            sum_columns = matrix.sum(axis=1)

            # get the index of the lowest sum
            lowest_sum_index = sum_columns.idxmin()

            # add the lowest sum to the sequence
            sequence.append(lowest_sum_index)

            # drop the row and column
            matrix = matrix.drop(lowest_sum_index, axis=0)
            matrix = matrix.drop(lowest_sum_index, axis=1)

        return pd.DataFrame({"answers": [",".join(sequence)]})

    def get_rel_score_mult(self, sequences: pd.DataFrame) -> float:

        # get the score of the suggested sequence for each row in the dataframe in the column "answers"
        scores = []
        n_failed = 0
        n_total = []
        n_violations = []
        for i, row in sequences.iterrows():
            if not self.sanity_check_row(row):
                n_failed += 1
                # scores.append(0)
                continue

            # turn the row into a dataframe
            row = pd.DataFrame(row).T
            score_abs, _, n_violations_loc, n_total_loc = self.get_score(row)
            # max_score = self.get_max_score_subset(row)

            # in gt matrix: make > 0.5 to 1 and <= 0.5 to 0

            # score = score_abs / max_score
            score = np.log(score_abs)

            if score > 1.02:
                raise ValueError(f"Score is larger than 1: {score}")

            scores.append(score)
            n_violations.append(n_violations_loc)
            n_total.append(n_total_loc)

        scores = pd.Series(scores)

        return scores.mean(), n_failed, n_violations, n_total

    def get_rel_score_mult_mat(
        self, matrix: pd.DataFrame
    ) -> tuple[float, int, list, list]:
        """Get the relative score for a matrix instead of a sequence.

        Args:
            matrix (pd.DataFrame): _description_

        Raises:
            ValueError: _description_

        Returns:
            tuple[float, int, list, list]: _description_
        """
        
        if self.hope_mapping_path:
            # map the column and row names to the alias names
            new_index = []
            for idx in matrix.index:
                if idx in self.hope_mapping and self.hope_mapping[idx]["alias_name"]:
                    new_index.append(self.hope_mapping[idx]["alias_name"])
                else:
                    new_index.append(idx)
            matrix.index = new_index

            new_columns = []
            for col in matrix.columns:
                if col in self.hope_mapping and self.hope_mapping[col]["alias_name"]:
                    new_columns.append(self.hope_mapping[col]["alias_name"])
                else:
                    new_columns.append(col)
            matrix.columns = new_columns

            # Sum duplicate rows and columns after renaming
            matrix = matrix.groupby(matrix.index).sum()  # Sum duplicate rows
            # Sum duplicate columns by transpose, group, sum, transpose
            matrix = matrix.T.groupby(matrix.T.index).sum().T

        # get the score of the suggested sequence for each row in the dataframe
        n_failed = 0
        n_total = []
        n_violations = []

        # assert all columns and rows are in the gt matrix
        not_present_keys = set(matrix.index).difference(set(self.gt_matrix.index))
        if len(not_present_keys) > 0:
            raise ValueError(
                f"Keys {not_present_keys} not in gt matrix!"
                f" Available keys: {self.gt_matrix.index.tolist()}"
                )

        score_abs, _, n_violations_loc, n_total_loc = self.get_score(
            matrix, is_matrix=True
        )
        # max_score = self.get_max_score_subset(row)

        # in gt matrix: make > 0.5 to 1 and <= 0.5 to 0

        # score = score_abs / max_score
        score = np.log(score_abs)

        if score > 1.02:
            raise ValueError(f"Score is larger than 1: {score}")

        n_violations.append(n_violations_loc)
        n_total.append(n_total_loc)

        return score, n_failed, n_violations, n_total

    def sanity_check_row(self, row: pd.DataFrame):
        # check, if there are at least 3 usable groceries in the row
        groceries = row["answers"].split(",")
        # groceries = [grocery for grocery in groceries if grocery in self.gt_matrix.index]

        g_cnt = 0

        for idx in self.gt_matrix.index:
            for grocery in groceries:
                if grocery.lower() in idx.lower():
                    g_cnt += 1
                else:
                    pass

        if len(groceries) < 3:
            return False
        else:
            return True

    def get_rel_score(self, sequence: pd.DataFrame) -> float:
        score, subset_of_groceries_utilized_in_answer, n_violations = (
            self.get_score(sequence)
        )
        max_score = self.get_max_score(self.gt_matrix)
        return score / max_score

    def write_matrix(self):
        # write the matrix to a file as xlsx
        cache_dir = get_cache_dir()
        matrix_file = cache_dir / "comparison_matrix.xlsx"
        self.gt_matrix.to_excel(matrix_file)

    def write_answers(self, answers_list: list):
        # write the answers to a json file
        cache_dir = get_cache_dir()
        answers_file = cache_dir / "groceries_from_poll.json"

        answers_list = {
            "groceries": answers_list,
            "comment": "This file is automatically generated when importing the poll results.",
        }

        with open(answers_file, "w") as f:
            json.dump(answers_list, f, indent=4)

    @staticmethod
    def list_to_df(sequence: list) -> str:
        # convert the list to a string
        sequence = ",".join(sequence)

        # make a dataframe with one entry which is the sequence as a string
        df = pd.DataFrame({"answers": [sequence]})

        return df

    def list_of_lists_to_df(self, sequences: list[list[str]]):
        dfs = []
        for seq in sequences:
            df = self.list_to_df(seq)
            dfs.append(df)

        return pd.concat(dfs)


def main():
    poll = PollEvaluator(csv_answer_file=Path("responses.csv"))

    # writing the matrix for debugging purposes
    poll.write_matrix()

    # this is the suggested by chat gpt
    gpt_suggested_sequence = [
        "Canned Beans",
        "Canned Corn",
        "Canned Peas",
        "Glass Beer Bottle",
        "1.5L Plastic Water Bottle",
        "Energy Drink Can",
        "Noodles in Plastic Bag",
        "Frozen Spinach",
        "Cheese",
        "Joghurt Cup",
        "Onions",
        "Bell Pepper",
        "Cucumber",
        "Lettuce",
        "Mushrooms",
        "Tomatoes",
        "Apples",
        "Bananas",
        "Strawberries",
        "Eggs",
    ]

    # Prompt: In which sequence would you load these items in a shopping bag? assume the shopping bag is narrow, so each item would go strictly on top of  he previous item. make sure that delicate items are higher. soft fruits for example might get squished. Also be careful abt liquid in delicate lastic backaging so thhat the packaging does not burst and make the liquid spill and make a mess. start your answer with lets think step by step.
    gpt_suggested_sequence = [
        "Canned Beans",
        "Canned Corn",
        "Canned Peas",
        "1.5L Plastic Water Bottle",
        "Glass Beer Bottle",
        "Energy Drink Can",
        "Frozen Spinach",
        "Noodles in Plastic Bag",
        "Onions",
        "Cucumber",
        "Bell Pepper",
        "Joghurt Cup",
        "Cheese",
        "Lettuce",
        "Mushrooms",
        "Tomatoes",
        "Apples",
        "Bananas",
        "Strawberries",
        "Eggs",
    ]

    deep_seek_generated_sequence = [
        "Canned Corn",
        "Canned Beans",
        "Canned Peas",
        "1.5L Plastic Water Bottle",
        "Energy Drink Can",
        "Glass Beer Bottle",
        "Frozen Spinach",
        "Onions",
        "Bell Pepper",
        "Cucumber",
        "Apples",
        "Tomatoes",
        "Lettuce",
        "Mushrooms",
        "Bananas",
        "Eggs",
        "Strawberries",
        "Joghurt Cup",
        "Cheese",
        "Noodles in Plastic Bag",
    ]

    # gpt_suggested_sequence = deep_seek_generated_sequence

    # check if we can get 100% score with the perfect sequence.
    # This is a basic sanity check
    perfect_sequence = poll.matrix_to_sequence(poll.gt_matrix)["answers"][0]
    perfect_sequence = perfect_sequence.split(",")
    perfect_score, _, _, _ = poll.get_score(
        pd.DataFrame({"answers": [",".join(perfect_sequence)]})
    )
    assert (
        perfect_score > 0.99
    ), f"The score of the perfect sequence is not close to 100%, but {perfect_score}."

    # print the perfect sequence
    print("")
    print(f"Perfect sequence: \n{','.join(perfect_sequence)}")

    # get the sequence as a dataframe
    df = poll.list_to_df(gpt_suggested_sequence)

    # get the score of the suggested sequence
    score, _, _ = poll.get_score(df)
    max_score = poll.get_max_score(poll.gt_matrix)
    rel_score = poll.get_rel_score(df)

    # make report
    print("")
    print(f"Score: {score}")
    print(f"Max Score: {max_score}")
    print(f"Relative Score: {rel_score}")


if __name__ == "__main__":
    main()
