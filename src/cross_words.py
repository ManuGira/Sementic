import dataclasses
import enum
import numpy as np
import common as cmn
import coordinatus as co

class Orientation(enum.Enum):
    HORIZONTAL = 'H'
    VERTICAL = 'V'

class CellState(enum.Enum):
    BLOCKED = 0
    FILLED = 1
    AVAILABLE = 2
    AVAILABLE_HORIZONTAL = 3
    AVAILABLE_VERTICAL = 4

@dataclasses.dataclass
class Letter:
    pos_xy: co.Point
    orientation: Orientation
    char: str = ""


class CrossWord:
    def __init__(self, word: str, pos_xy: co.Point, orientation: Orientation):
        self.word = word
        self.orientation = orientation
        assert isinstance(pos_xy, co.Point), f"pos_xy must be a coordinatus Point, got {type(pos_xy)}"
        self.pos_xy = pos_xy

        # Set up local coordinate system for the word
        tx, ty = pos_xy.coords.round().astype(int)
        sy = 1
        angle_rad = 0.0
        if self.orientation == Orientation.VERTICAL:
            sy=-1
            angle_rad=np.pi/2

        self.local_frame = co.create_frame(None, tx=tx, ty=ty, angle_rad=angle_rad, sy=sy)


    def board_to_local_coordinates(self, board_pos_xy: tuple[int, int]) -> tuple[int, int]:
        """Convert global grid coordinates to local word coordinates."""
        pos = co.Point(board_pos_xy)
        pos = pos.relative_to(self.local_frame)
        local_pos_xy = pos.coords.round().astype(int).tolist()
        return local_pos_xy.tolist()

    def local_to_board_coordinates(self, local_pos_xy: tuple[int, int]) -> tuple[int, int]:
        """Convert local word coordinates to global grid coordinates."""
        pos = co.Point(local_pos_xy, frame=self.local_frame)
        pos = pos.to_absolute()
        board_pos_xy = pos.coords.round().astype(int).tolist()
        return board_pos_xy

    def __eq__(self, other: 'CrossWord'):
        if not isinstance(other, CrossWord):
            return NotImplemented
        return (self.word == other.word and
                self.pos_xy == other.pos_xy and
                self.orientation == other.orientation)

    def char_at_grid_pos(self, pos_xy: co.Point) -> tuple[CellState, str]:
        """Get character at given position if it exists in the word placement."""
        x, y = pos_xy.relative_to(self.local_frame).coords.round().astype(int)

        y_distance = abs(y)
        if x < 0:
            x_distance = -x
        elif x >= len(self.word):
            x_distance = x - len(self.word) + 1
        else:
            x_distance = 0
        distance = y_distance + x_distance
        
        if distance > 1:
            return CellState.AVAILABLE, ""
        
        if distance == 1:
            if y_distance == 0:
                return CellState.BLOCKED, ""

            match self.orientation:
                case Orientation.HORIZONTAL:
                    return CellState.AVAILABLE_VERTICAL, ""
                case Orientation.VERTICAL:
                    return CellState.AVAILABLE_HORIZONTAL, ""

        return CellState.FILLED, self.word[x]

    def letters(self) -> list[Letter]:
        """Get all letters with their positions and orientations."""
        letters = []
        for idx, char in enumerate(self.word):
            letters.append(
                Letter(
                    co.Point([idx, 0], self.local_frame).to_absolute(),
                    self.orientation,
                    char,
                ))

        return letters

    def get_pre_padding(self) -> Letter:
        board_pos = co.Point((-1, 0), self.local_frame).to_absolute()
        return Letter(board_pos, self.orientation, "0")

    def get_post_padding(self) -> Letter:
        board_pos = co.Point((len(self.word), 0), self.local_frame).to_absolute()
        return Letter(board_pos, self.orientation, "0")

        
    def find_cross_points(self, other_word: str) -> list['CrossWord']:
        """Find cross points between this word and another word."""
        cross_points = []
        other_orientation = Orientation.VERTICAL if self.orientation == Orientation.HORIZONTAL else Orientation.HORIZONTAL

        for i, char1 in enumerate(self.word):
            for j, char2 in enumerate(other_word):
                if char1 == char2:
                    board_pos = co.Point([i, -j], self.local_frame).to_absolute()
                    cross_points.append(CrossWord(other_word, board_pos, other_orientation))
        return cross_points
    
        
class CrossWordsBoard:
    def __init__(self):
        self.placements: list[CrossWord] = []

    def is_valid_placement(self, new_word: CrossWord) -> bool:
        """Check if a crossword placement is valid on the board."""

        letters = [new_word.get_pre_padding()] + new_word.letters() + [new_word.get_post_padding()]

        for letter in letters:
            for crossword in self.placements:
                crossword_state, crossword_char = crossword.char_at_grid_pos(letter.pos_xy)

                if crossword_state == CellState.BLOCKED:
                    return False

                if crossword_state == CellState.FILLED and crossword_char != letter.char:
                    return False

                if crossword_state == CellState.AVAILABLE_HORIZONTAL and letter.orientation != Orientation.HORIZONTAL:
                    return False

                if crossword_state == CellState.AVAILABLE_VERTICAL and letter.orientation != Orientation.VERTICAL:
                    return False

        return True

    def add_word(self, word: CrossWord):
        self.placements.append(word)

    def compute_new_word_placements(self, word: str):
        if len(self.placements) == 0:
            return [CrossWord(word, co.Point([0, 0]), Orientation.HORIZONTAL)]

        valid_crossword_positions: list[CrossWord] = []
        for crossword in self.placements:
            valid_crossword_positions += crossword.find_cross_points(word)

        # filter valid placements
        valid_crossword_positions = [cp for cp in valid_crossword_positions if self.is_valid_placement(cp)]
        return valid_crossword_positions


    def get_char(self, pos_xy: co.Point) -> str:
        """Get character at given position from any word placement."""
        for placement in self.placements:
            cellstate, char = placement.char_at_grid_pos(pos_xy)
            if cellstate == CellState.FILLED:
                return char
        return ""


    def generate_board(self) -> list[list[str]]:
        """Display the current state of the board."""
        if not self.placements:
            print("Board is empty.")
            return [["."]]

        letters = [letter for placement in self.placements for letter in placement.letters()]

        min_x = int(round(min(letter.pos_xy[0] for letter in letters)))
        max_x = int(round(max(letter.pos_xy[0] for letter in letters)))
        min_y = int(round(min(letter.pos_xy[1] for letter in letters)))
        max_y = int(round(max(letter.pos_xy[1] for letter in letters)))

        board_frame = co.create_frame(None, tx=min_x, ty=min_y)
        width = max_x - min_x + 1
        height = max_y - min_y + 1
        board_array = np.full((height, width), ".", dtype=str)

        for letter in letters:
            pos_xy = letter.pos_xy.relative_to(board_frame)
            x, y = pos_xy.coords.round().astype(int)
            board_array[y, x] = letter.char

        return board_array.tolist()


def load_words():
    MODEL_PATH = "frWac_non_lem_no_postag_no_phrase_200_cbow_cut100.bin"  # TODO: Update with actual path

    # Step 1: Load word2vec model
    print("Loading word2vec model...")
    model = cmn.load_model(MODEL_PATH)

    print("Loading frequent words...")
    frequent_words = cmn.load_most_frequent_words(model=model)

    # filter out words shorter than 3 characters
    frequent_words = [word for word in frequent_words if len(word) >= 3]

    # Choose a root word at random from the frequent words
    root_word = np.random.choice(frequent_words)

    # sort all words by their similarity to the root word
    print("Sorting words by similarity to root word:", root_word)
    sorted_words = sorted(frequent_words, key=lambda w: model.similarity(root_word, w), reverse=True)
    return sorted_words

def main():
    board = CrossWordsBoard()
    # words = load_words()
    words = ["école", "primaire", "scolaire", "élémentaire", "collège", "classe", "élève", "éducation", "instituteur", "lycée", "secondaire", "enseigner", "enseignant", "rentrée", "enseignement", "professeur", "cycle", "apprendre", "civique", "fréquenter", "sixième", "enlever", "apprentissage", "établissement", "réussite", "enfant", "enfance", "inspecteur", "discipline", "orientation", "citoyenneté", "institution", "instruire", "degré", "adulte", "difficulté", "supérieur", "adolescent", "cours", "année", "dès", "rénovation", "prioritaire", "gamin", "technologie", "initier", "inspection", "précoce", "échec", "collègue", "famille", "amener", "universitaire", "grand-père", "grand-mère", "fiche", "lecture", "culture", "jeune", "municipalité", "métier", "cirque", "camarade", "progressivement", "âge", "intégration", "autrefois", "réussir", "copain", "préparer", "intégrer", "atelier", "instruction", "grandir", "maison", "pratique", "former", "mère", "dans", "cité", "travailler", "sou", "tante", "garçon", "entrée", "parent", "aider", "excellence", "maître", "présentement", "artistique", "autonomie", "jeunesse", "ingénieur", "approprier", "tôt", "maternel", "encadrement", "leçon", "écriture"]

    for word in words:
        # Compute candidate placements for the new word
        candidate_positions: list[CrossWord] = board.compute_new_word_placements(word)
        print(f"Candidates for '{word}':")
        for candidate in candidate_positions:
            print("\t", candidate)
        if not candidate_positions:
            print(f"No valid placements for '{word}'. Skipping.")
            continue

        if candidate_positions:
            # Group by candidate duplicates. choose in priority candidates with most duplicates.
            # This favors placements that cross multiple words, making the game easier to solve
            candidate_count: dict[int, list[CrossWord]] = {}
            for candidate in candidate_positions:
                count = sum(1 for other in candidate_positions if other == candidate)
                if count not in candidate_count:
                    candidate_count[count] = []
                candidate_count[count].append(candidate)

            max_count = max(candidate_count.keys())
            candidate_positions = candidate_count[max_count]

        new_crossword = np.random.choice(candidate_positions) if candidate_positions else None
        board.add_word(new_crossword)

        if len(board.placements) == 10:
            break

    print("Final board:")
    for row in board.generate_board():
        print(" ".join(row))

def main2():
    board = CrossWordsBoard()
    word1 = CrossWord("chat", co.Point([0, 0]), Orientation.HORIZONTAL)
    board.add_word(word1)

    word2 = CrossWord("table", co.Point([-1, -1]), Orientation.VERTICAL)
    if board.is_valid_placement(word2):
        board.add_word(word2)

    board.generate_board()

if __name__ == "__main__":
    main()