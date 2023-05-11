import heapq
import copy
import numpy as np

def get_manhattan_distance(from_state, to_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: implement this function. This function will not be tested directly by the grader. 

    INPUT: 
        Two states (if second state is omitted then it is assumed that it is the goal state)

    RETURNS:
        A scalar that is the sum of Manhattan distances for all tiles.
    """
    index_dictionary = {0: (1, 1), 1: (1, 2), 2: (1, 3), 3: (2, 1), 4: (2, 2), 5: (2, 3), 6: (3, 1), 7: (3, 2),
                        8: (3, 3)}

    manhattan_distance = 0
    for i_from_tile in range(len(from_state)):
        curr_from_tile = from_state[i_from_tile]
        if curr_from_tile != 0:
            for i_to_tile in range(len(to_state)):
                curr_to_tile = to_state[i_to_tile]
                if curr_to_tile == curr_from_tile:
                    from_position = index_dictionary[i_from_tile]
                    to_position = index_dictionary[i_to_tile]
                    x_diff = abs(from_position[0] - to_position[0])
                    y_diff = abs(from_position[1] - to_position[1])
                    manhattan_distance += x_diff + y_diff

    return manhattan_distance


def print_succ(state):
    """
    TODO: This is based on get_succ function below, so should implement that function.

    INPUT: 
        A state (list of length 9)

    WHAT IT DOES:
        Prints the list of all the valid successors in the puzzle. 
    """
    succ_states = get_succ(state)
    for succ_state in succ_states:
        print(succ_state, "h={}".format(get_manhattan_distance(succ_state)))


def get_succ(state):
    """
    TODO: implement this function.

    INPUT: 
        A state (list of length 9)

    RETURNS:
        A list of all the valid successors in the puzzle (don't forget to sort the result as done below). 
    """
    index_dictionary = {0: [1, 1], 1: [1, 2], 2: [1, 3], 3: [2, 1], 4: [2, 2], 5: [2, 3], 6: [3, 1], 7: [3, 2],
                        8: [3, 3]}
    # Step 1: Get all 0 positions
    zero_position_list = list()
    zero_indices = list()
    for i_state in range(len(state)):
        if state[i_state] == 0:
            zero_position_list.append(copy.deepcopy(index_dictionary[i_state]))
            zero_indices.append(i_state)

    # Step 2: Get all unique positions of distance one from them
    # Creates a dictionary of zero index to position of adjacent tile to swap
    adjacent_positions = dict()
    for i_zero in range(len(zero_position_list)):
        adjacent_positions[zero_indices[i_zero]] = list()
        # x + 1
        trial_position = copy.deepcopy(zero_position_list[i_zero])
        trial_position[0] = trial_position[0] + 1
        if trial_position in index_dictionary.values() and trial_position not in zero_position_list:
            adjacent_positions[zero_indices[i_zero]].append(trial_position)
        # x - 1
        trial_position = copy.deepcopy(zero_position_list[i_zero])
        trial_position[0] = trial_position[0] - 1
        if trial_position in index_dictionary.values() and trial_position not in zero_position_list:
            adjacent_positions[zero_indices[i_zero]].append(trial_position)
        # y + 1
        trial_position = copy.deepcopy(zero_position_list[i_zero])
        trial_position[1] = trial_position[1] + 1
        if trial_position in index_dictionary.values() and trial_position not in zero_position_list:
            adjacent_positions[zero_indices[i_zero]].append(trial_position)
        # y - 1
        trial_position = copy.deepcopy(zero_position_list[i_zero])
        trial_position[1] = trial_position[1] - 1
        if trial_position in index_dictionary.values() and trial_position not in zero_position_list:
            adjacent_positions[zero_indices[i_zero]].append(trial_position)

    # Step 3: Find indices of state corresponding to these positions
    # Creates a dictionary of zero index to corresponding tile indices to swap
    adjacent_indices = dict()
    for zero_index in adjacent_positions.keys():
        adjacent_indices[zero_index] = list()
        adjacent_position_list = adjacent_positions[zero_index]
        for curr_position in adjacent_position_list:
            for index_index in index_dictionary.keys():
                if index_dictionary[index_index] == curr_position:
                    adjacent_indices[zero_index].append(index_index)

    # Step 4: Swap indices with 0 one by one generating a new succ_state each time
    succ_states = list()
    for zero_index in adjacent_indices.keys():
        for position_index in adjacent_indices[zero_index]:
            moldable_state = copy.deepcopy(state)

            temp_tile = moldable_state[zero_index]
            moldable_state[zero_index] = moldable_state[position_index]
            moldable_state[position_index] = temp_tile

            succ_states.append(moldable_state)

    return sorted(succ_states)


def solve(state, goal_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: Implement the A* algorithm here.

    INPUT: 
        An initial state (list of length 9)

    WHAT IT SHOULD DO:
        Prints a path of configurations from initial state to goal state along  h values, number of moves, and max queue number in the format specified in the pdf.
    """
    # Format to put into priority queue:
    # 0th index is the A* Score -- i.e. f = h + g
    # 1st index is its actual state
    # 2nd index is g -- total count of past moves
    # 3rd index is a list of its trace back states

    output_tile = None
    traceback_states = 0
    max_queue_length = 0

    open = []
    closed = list()
    heapq.heappush(open, (get_manhattan_distance(state), state, 0, list()))
    while len(open) > 0:
        if len(open) > max_queue_length:
            max_queue_length = len(open)

        curr_tile = heapq.heappop(open)

        curr_state = curr_tile[1]
        if curr_state == goal_state:
            # We've reached the goal!!
            output_tile = curr_tile
            break

        closed.append(curr_tile)

        curr_g = curr_tile[2]
        curr_traceback = copy.deepcopy(curr_tile[3])
        curr_traceback.append(curr_state)

        # Expanding to new nodes
        successors = get_succ(curr_state)
        for succ in successors:
            # First, check if the successor's already in open or closed
            add_boolean = 1
            old_g = np.inf
            for c in closed:
                if succ == c[1]:
                    add_boolean = 0
                    old_g = c[2]
                    break
            #for o in open:
            #    if succ == o[1]:
            #        add_boolean = 0
            #        old_g = o[2]
            #        break

            # If it's not in open or closed, calculate h and f and add it to open
            if add_boolean == 1:
                h = get_manhattan_distance(succ)
                f = h + curr_g + 1
                heapq.heappush(open, (f, succ, curr_g + 1, curr_traceback))
            # If it is in open or closed, check whether g of successor less than the old
            # If so, add the successor tile to open with the correct traceback
            elif add_boolean == 0:
                if (curr_g + 1) < old_g:
                    h = get_manhattan_distance(succ)
                    f = h + curr_g + 1
                    heapq.heappush(open, (f, succ, curr_g + 1, curr_traceback))

    # A* Finished -- now print out results
    if output_tile is None:
        return "A* Search Failed"
    else:
        moves = 0

        # Print traceback
        traceback_states = output_tile[3]
        for trace_state in traceback_states:
            h = get_manhattan_distance(trace_state)
            print(str(trace_state) + " h=" + str(h) + " moves: " + str(moves))
            moves += 1

        # Print final state
        output_state = output_tile[1]
        h = get_manhattan_distance(output_state)
        print(str(output_state) + " h=" + str(h) + " moves: " + str(moves))
        moves += 1

        # Print max queue length
        print("Max queue length:", max_queue_length)


if __name__ == "__main__":
    """
    Feel free to write your own test code here to examine the correctness of your functions. 
    Note that this part will not be graded.
    """
    print_succ([2, 5, 1, 4, 0, 6, 7, 0, 3])
    print()

    print(get_manhattan_distance([2, 5, 1, 4, 0, 6, 7, 0, 3]))
    print()

    solve([1, 7, 0, 6, 3, 2, 0, 4, 5])
    print()



