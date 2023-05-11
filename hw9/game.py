import random
import copy
import numpy as np


class TeekoPlayer:
    """ An object representation for an AI game player for the game Teeko.
    """
    board = [[' ' for j in range(5)] for i in range(5)]
    pieces = ['b', 'r']

    def __init__(self):
        """ Initializes a TeekoPlayer object by randomly selecting red or black as its
        piece color.
        """
        self.my_piece = random.choice(self.pieces)
        self.opp = self.pieces[0] if self.my_piece == self.pieces[1] else self.pieces[1]

    def succ(self, state, drop_phase):
        # If we are still in the drop state
        if drop_phase == 1:
            successors = list()
            for i_row in range(5):
                for i_col in range(5):
                    poss_succ = copy.deepcopy(state)
                    if poss_succ[i_row][i_col] == ' ':
                        poss_succ[i_row][i_col] = self.my_piece
                        successors.append(poss_succ)
            return successors
        # If we are in the move state
        else:
            successors = list()
            # First, identify the positions where our pieces are
            for i_row in range(5):
                for i_col in range(5):
                    if state[i_row][i_col] == self.my_piece:
                        # Now checking all 8 possible adjacent positions to where our piece currently is
                        for delta_row in range(-1, 2):
                            for delta_col in range(-1, 2):
                                if not (delta_row == 0 and delta_col == 0):
                                    new_row = i_row + delta_row
                                    new_col = i_col + delta_col
                                    # First check that row and column are in range
                                    # Then that the position is open
                                    if 0 <= new_row <= 4 and 0 <= new_col <= 4 and state[new_row][new_col] == ' ':
                                        poss_succ = copy.deepcopy(state)
                                        poss_succ[i_row][i_col] = ' '
                                        poss_succ[new_row][new_col] = self.my_piece
                                        successors.append(poss_succ)
            return successors

    # This is looking at only the state itself and how preferable it is!!
    def heuristic_game_value(self, state, drop_phase):
        # Take the absolute value of it in make_move(), not here!
        curr_state_game_value = self.game_value(state)
        if curr_state_game_value != 0:
            return curr_state_game_value
        else:
            # Average across list at the end
            heuristic_counter = list()
            # TODO: Checking for horizontal connections
            for i_row in range(5):
                for i_col in range(5):
                    if state[i_row][i_col] == self.my_piece:
                        if i_col + 2 <= 4 and state[i_row][i_col] == state[i_row][i_col + 1] == state[i_row][i_col + 2]:
                            heuristic_counter.append(0.75)
                        if i_col + 1 <= 4 and state[i_row][i_col] == state[i_row][i_col + 1]:
                            heuristic_counter.append(0.5)
                    if state[i_row][i_col] == self.opp:
                        if i_col + 2 <= 4 and state[i_row][i_col] == state[i_row][i_col + 1] == state[i_row][i_col + 2]:
                            heuristic_counter.append(-0.80)
                        if i_col + 1 <= 4 and state[i_row][i_col] == state[i_row][i_col + 1]:
                            heuristic_counter.append(-0.55)

            # TODO: Checking for vertical connections
            for i_row in range(5):
                for i_col in range(5):
                    if state[i_row][i_col] == self.my_piece:
                        if i_row + 2 <= 4 and state[i_row][i_col] == state[i_row + 1][i_col] == state[i_row + 2][i_col]:
                            heuristic_counter.append(0.75)
                        if i_row + 1 <= 4 and state[i_row][i_col] == state[i_row + 1][i_col]:
                            heuristic_counter.append(0.5)
                    if state[i_row][i_col] == self.opp:
                        if i_row + 2 <= 4 and state[i_row][i_col] == state[i_row + 1][i_col] == state[i_row + 2][i_col]:
                            heuristic_counter.append(-0.80)
                        if i_row + 1 <= 4 and state[i_row][i_col] == state[i_row + 1][i_col]:
                            heuristic_counter.append(-0.55)
            # TODO: Checking for \ connections
            for i_row in range(5):
                for i_col in range(5):
                    if state[i_row][i_col] == self.my_piece:
                        if i_row + 2 <= 4 and i_col + 2 <= 4 and state[i_row][i_col] == state[i_row + 1][i_col + 1] == \
                                state[i_row + 2][i_col + 2]:
                            heuristic_counter.append(0.66)
                        if i_row + 2 <= 4 and i_col + 2 <= 4 and state[i_row][i_col] == state[i_row + 1][i_col + 1]:
                            heuristic_counter.append(0.33)
                    if state[i_row][i_col] == self.opp:
                        if i_row + 2 <= 4 and i_col + 2 <= 4 and state[i_row][i_col] == state[i_row + 1][i_col + 1] == \
                                state[i_row + 2][i_col + 2]:
                            heuristic_counter.append(-0.71)
                        if i_row + 2 <= 4 and i_col + 2 <= 4 and state[i_row][i_col] == state[i_row + 1][i_col + 1]:
                            heuristic_counter.append(-0.38)
            # TODO: Checking for / connections
            for i_row in range(5):
                for i_col in range(5):
                    if state[i_row][i_col] == self.my_piece:
                        if i_row + 2 <= 4 and i_col + 2 <= 4 and state[i_row][i_col] == state[i_row + 1][i_col - 1] == \
                                state[i_row + 2][i_col - 2]:
                            heuristic_counter.append(0.66)
                        if i_row + 2 <= 4 and i_col + 2 <= 4 and state[i_row][i_col] == state[i_row + 1][i_col - 1]:
                            heuristic_counter.append(0.33)
                    if state[i_row][i_col] == self.opp:
                        if i_row + 2 <= 4 and 0 <= i_col - 2 and state[i_row][i_col] == state[i_row + 1][i_col - 1] == \
                                state[i_row + 2][i_col - 2]:
                            heuristic_counter.append(-0.71)
                        if i_row + 2 <= 4 and 0 <= i_col - 2 and state[i_row][i_col] == state[i_row + 1][i_col - 1]:
                            heuristic_counter.append(-0.38)
            # TODO: Checking for box connections
            for i_row in range(5):
                for i_col in range(5):
                    if state[i_row][i_col] == self.my_piece and i_row + 1 <= 4 and i_col + 1 <= 4:
                        box_counter = 0
                        if state[i_row][i_col] == self.my_piece:
                            box_counter += 1
                        if state[i_row][i_col + 1] == self.my_piece:
                            box_counter += 1
                        if state[i_row + 1][i_col] == self.my_piece:
                            box_counter += 1
                        if state[i_row + 1][i_col + 1] == self.my_piece:
                            box_counter += 1
                        if box_counter == 3:
                            heuristic_counter.append(0.6)
                        elif box_counter == 2:
                            heuristic_counter.append(0.3)
                    if state[i_row][i_col] == self.opp and i_row + 1 <= 4 and i_col + 1 <= 4:
                        box_counter = 0
                        if state[i_row][i_col] == self.opp:
                            box_counter += 1
                        if state[i_row][i_col + 1] == self.opp:
                            box_counter += 1
                        if state[i_row + 1][i_col] == self.opp:
                            box_counter += 1
                        if state[i_row + 1][i_col + 1] == self.opp:
                            box_counter += 1
                        if box_counter == 3:
                            heuristic_counter.append(-0.65)
                        elif box_counter == 2:
                            heuristic_counter.append(-0.35)
            if len(heuristic_counter) == 0:
                return 0
            else:
                return sum(heuristic_counter) / len(heuristic_counter)

    def max_value(self, state, drop_phase, depth):
        # Check if drop stage is now over for successors:
        if drop_phase == True:
            my_counter = 0
            for i_row in range(5):
                for i_col in range(5):
                    if state[i_row][i_col] == self.my_piece:
                        my_counter += 1
            if my_counter == 4:
                drop_phase = False
        # Do max half of minimax:
        max_depth = 15
        if self.game_value(state) != 0:
            return self.game_value(state)
        elif depth == max_depth:
            return self.heuristic_game_value(state, drop_phase)
        else:
            alpha = -np.inf
            successors = self.succ(state, drop_phase)
            for s in successors:
                return max(alpha, self.min_value(s, drop_phase, depth + 1))

    def min_value(self, state, drop_phase, depth):
        # Check if drop state is now over for successors:
        if drop_phase == True:
            my_counter = 0
            for i_row in range(5):
                for i_col in range(5):
                    if state[i_row][i_col] == self.my_piece:
                        my_counter += 1
            if my_counter == 4:
                drop_phase = False
        # Do min half of minimax:
        max_depth = 15
        if self.game_value(state) != 0:
            return self.game_value(state)
        elif depth == max_depth:
            return self.heuristic_game_value(state, drop_phase)
        else:
            alpha = np.inf
            successors = self.succ(state, drop_phase)
            for s in successors:
                return min(alpha, self.max_value(s, drop_phase, depth + 1))

    def make_move(self, state):
        """ Selects a (row, col) space for the next move. You may assume that whenever
        this function is called, it is this player's turn to move.

        Args:
            state (list of lists): should be the current state of the game as saved in
                this TeekoPlayer object. Note that this is NOT assumed to be a copy of
                the game state and should NOT be modified within this method (use
                place_piece() instead). Any modifications (e.g. to generate successors)
                should be done on a deep copy of the state.

                In the "drop phase", the state will contain less than 8 elements which
                are not ' ' (a single space character).

        Return:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

        Note that without drop phase behavior, the AI will just keep placing new markers
            and will eventually take over the board. This is not a valid strategy and
            will earn you no points.
        """
        # TODO: detect drop phase
        # Count across all entries to see if there are 4 of my pieces
        my_piece_counter = 0
        for i_row in range(5):
            for i_col in range(5):
                if state[i_row][i_col] == self.my_piece:
                    my_piece_counter += 1
        if my_piece_counter == 4:
            drop_phase = False
        else:
            drop_phase = True

        # TODO: implement a minimax algorithm to play better
        # minimax returns an estimate of a state's utility value

        #(row, col) = (random.randint(0, 4), random.randint(0, 4)) -- old random selection
        #while not state[row][col] == ' ':
        #    (row, col) = (random.randint(0, 4), random.randint(0, 4))
        move = []

        # Getting the best successor
        successors = self.succ(state, drop_phase)
        successor_utility = np.zeros(len(successors))
        for i_succ in range(len(successors)):
            successor_utility[i_succ] = self.max_value(successors[i_succ], drop_phase, 0)
        best_successor_index = np.argmax(successor_utility)
        best_successor = successors[best_successor_index]

        # Finding the difference in the best successor:
        move_list = list()
        move_list.append((-1, -1))
        # If in the drop phase, only return new position
        if drop_phase:
            for i_row in range(5):
                for i_col in range(5):
                    if state[i_row][i_col] != best_successor[i_row][i_col]:
                        move_list[0] = (i_row, i_col)
        # If not in the drop phase, return new position and source position
        else:
            for i_row in range(5):
                for i_col in range(5):
                    if state[i_row][i_col] != best_successor[i_row][i_col]:
                        # If it is the new position, set the first index to it
                        if best_successor[i_row][i_col] == self.my_piece:
                            move_list[0] = (i_row, i_col)
                        # If it is the old position, add the source in the second index
                        if state[i_row][i_col] == self.my_piece:
                            move_list.append((i_row, i_col))

        #take into account when other team is blocking!!

        # ensure the destination (row,col) tuple is at the beginning of the move list
        move_list
        return move_list

    def opponent_move(self, move):
        """ Validates the opponent's next move against the internal board representation.
        You don't need to touch this code.

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.
        """
        # validate input
        if len(move) > 1:
            source_row = move[1][0]
            source_col = move[1][1]
            if source_row != None and self.board[source_row][source_col] != self.opp:
                self.print_board()
                print(move)
                raise Exception("You don't have a piece there!")
            if abs(source_row - move[0][0]) > 1 or abs(source_col - move[0][1]) > 1:
                self.print_board()
                print(move)
                raise Exception('Illegal move: Can only move to an adjacent space')
        if self.board[move[0][0]][move[0][1]] != ' ':
            raise Exception("Illegal move detected")
        # make move
        self.place_piece(move, self.opp)

    def place_piece(self, move, piece):
        """ Modifies the board representation using the specified move and piece

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

                This argument is assumed to have been validated before this method
                is called.
            piece (str): the piece ('b' or 'r') to place on the board
        """
        if len(move) > 1:
            self.board[move[1][0]][move[1][1]] = ' '
        self.board[move[0][0]][move[0][1]] = piece

    def print_board(self):
        """ Formatted printing for the board """
        for row in range(len(self.board)):
            line = str(row) + ": "
            for cell in self.board[row]:
                line += cell + " "
            print(line)
        print("   A B C D E")

    def game_value(self, state):
        """ Checks the current board status for a win condition

        Args:
        state (list of lists): either the current state of the game as saved in
            this TeekoPlayer object, or a generated successor state.

        Returns:
            int: 1 if this TeekoPlayer wins, -1 if the opponent wins, 0 if no winner

        TODO: complete checks for diagonal and box wins
        """
        # check horizontal wins
        for row in state:
            for i in range(2):
                if row[i] != ' ' and row[i] == row[i + 1] == row[i + 2] == row[i + 3]:
                    return 1 if row[i] == self.my_piece else -1

        # check vertical wins
        for col in range(5):
            for i in range(2):
                if state[i][col] != ' ' and state[i][col] == state[i + 1][col] == \
                        state[i + 2][col] == state[i + 3][col]:
                    return 1 if state[i][col] == self.my_piece else -1

        # TODO: check \ diagonal wins
        for i_row in range(2):
            for i_col in range(2):
                if state[i_row][i_col] != ' ' and state[i_row][i_col] == state[i_row + 1][i_col + 1] == \
                        state[i_row + 2][i_col + 2] == state[i_row + 3][i_col + 3]:
                    return 1 if state[i_row][i_col] == self.my_piece else -1

        # TODO: check / diagonal wins
        for i_row in range(2):
            for i_col in range(3, 5):
                if state[i_row][i_col] != ' ' and state[i_row][i_col] == state[i_row + 1][i_col - 1] == \
                        state[i_row + 2][i_col - 2] == state[i_row + 3][i_col - 3]:
                    return 1 if state[i_row][i_col] == self.my_piece else -1

        # TODO: check box wins
        for i_row in range(4):
            for i_col in range(4):
                if state[i_row][i_col] != ' ' and state[i_row][i_col] == state[i_row][i_col + 1] == \
                        state[i_row + 1][i_col] == state[i_row + 1][i_col + 1]:
                    return 1 if state[i_row][i_col] == self.my_piece else -1

        return 0  # no winner yet


############################################################################
#
# THE FOLLOWING CODE IS FOR SAMPLE GAMEPLAY ONLY
#
############################################################################
def main():
    print('Hello, this is Samaritan')
    ai = TeekoPlayer()
    piece_count = 0
    turn = 0

    # drop phase
    while piece_count < 8 and ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece + " moved at " + chr(move[0][1] + ord("A")) + str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp + "'s turn")
            while not move_made:
                player_move = input("Move (e.g. B3): ")
                while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
                    player_move = input("Move (e.g. B3): ")
                try:
                    ai.opponent_move([(int(player_move[1]), ord(player_move[0]) - ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        piece_count += 1
        turn += 1
        turn %= 2

    # move phase - can't have a winner until all 8 pieces are on the board
    while ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece + " moved from " + chr(move[1][1] + ord("A")) + str(move[1][0]))
            print("  to " + chr(move[0][1] + ord("A")) + str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp + "'s turn")
            while not move_made:
                move_from = input("Move from (e.g. B3): ")
                while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
                    move_from = input("Move from (e.g. B3): ")
                move_to = input("Move to (e.g. B3): ")
                while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
                    move_to = input("Move to (e.g. B3): ")
                try:
                    ai.opponent_move([(int(move_to[1]), ord(move_to[0]) - ord("A")),
                                      (int(move_from[1]), ord(move_from[0]) - ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        turn += 1
        turn %= 2

    ai.print_board()
    if ai.game_value(ai.board) == 1:
        print("AI wins! Game over.")
    else:
        print("You win! Game over.")


if __name__ == "__main__":
    main()
