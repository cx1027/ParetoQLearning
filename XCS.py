import numpy
import numpy.random
import itertools
from copy import deepcopy

"""
A class that represents the parameters of an NXCS system
"""
class parameters:
    def __init__(self):
        self.state_length = 5     #The number of bits in the state
        self.num_actions = 2      #The number of actions in this system
        self.theta_mna = 2        #The minimum number of elements in the match set
        self.initial_theta = 0.0              #The initial value of theta
        self.initial_prediction = 0.01        #The initial prediction value in classifiers
        self.initial_error = 0.01             #The initial error value in classifiers
        self.initial_fitness = 0.01           #The initial fitness value in classifiers
        self.p_hash = 0.3                     #The probability of generating a hash in a condition
        self.gamma = 0.71                     #The payoff decay rate
        self.alpha = 0.1
        self.beta = 0.2
        self.nu = 5
        self.N = 800                          #The maximum number of classifiers in the population
        self.e0 = 0.01                        #The minimum error value
        self.theta_del = 25                   #The experience level below which we don't delete classifiers
        self.delta = 0.1                      #The multiplier for the deletion vote of a classifier
        self.theta_sub = 20                   #The rate of subsumption
        self.theta_ga = 25                    #The rate of the genetic algorithm
        self.crossover_rate = 0.8
        self.mutation_rate = 0.04
        self.do_GA_subsumption = True
        self.do_action_set_subsumption = True
        self.rho0 = 1000

"""
A classifier in NXCS
"""
class classifier:
    global_id = 0 #A Globally unique identifier
    def __init__(self, parameters, state = None):
        self.id = classifier.global_id
        classifier.global_id = classifier.global_id + 1
        self.action = numpy.random.randint(0, parameters.num_actions)
        self.prediction = parameters.initial_prediction
        self.error = parameters.initial_error
        self.fitness = parameters.initial_fitness
        self.theta = parameters.initial_theta
        self.experience = 0
        self.time_stamp = 0
        self.average_size = 1
        self.numerosity = 1
        if state == None:
            self.condition = ''.join(['#' if numpy.random.rand() < parameters.p_hash else '0' if numpy.random.rand() > 0.5 else '1' for i in [0] * parameters.state_length])
        else:
            #Generate the condition from the state (if supplied)
            self.condition = ''.join(['#' if numpy.random.rand() < parameters.p_hash else state[i] for i in range(parameters.state_length)])

    def __str__(self):
        return "Classifier {0}: {1} = {2} Fitness: {3} Prediction: {4} Error: {5} Experience: {6} Theta: {7}".format(self.id, self.condition, self.action, self.fitness, self.prediction, self.error, self.experience, self.theta)

    """
       Mutates this classifier, changing the condition and action
       @param state - The state of the system to mutate around
       @param mutation_rate - The probability with which to mutate
       @param num_actions - The number of actions in the system
    """
    def _mutate(self, state, mutation_rate, num_actions):
        self.condition = ''.join([self.condition[i] if numpy.random.rand() > mutation_rate else state[i] if self.condition[i] == '#' else '#' for i in range(len(self.condition))])
        if numpy.random.rand() < mutation_rate:
            self.action = numpy.random.randint(0, num_actions)

    """
       Calculates the deletion vote for this classifier, that is, how much it thinks it should be deleted
       @param average_fitness - The average fitness in the current action set
       @param theta_del - See parameters above
       @param delta - See parameters above
    """
    def _delete_vote(self, average_fitness, theta_del, delta):
        vote = self.average_size * self.numerosity
        if self.experience > theta_del and self.fitness / self.numerosity < delta * average_fitness:
            return vote * average_fitness / (self.fitness / self.numerosity)
        else:
            return vote

    """
        Returns whether this classifier can subsume others
        @param theta_sub - See parameters above
        @param e0 - See parameters above
    """
    def _could_subsume(self, theta_sub, e0):
        return self.experience > theta_sub and self.error < e0

    """
        Returns whether this classifier is more general than another
        @param other - the classifier to check against
    """
    def _is_more_general(self, other):
        assert(len(self.condition) == len(other.condition)), "Invalid conditions"
        if len([i for i in self.condition if i == '#']) <= len([i for i in other.condition if i == '#']):
            return False

        return all(s == '#' or s == o for s, o in zip(self.condition, other.condition))

    """
        Returns whether this classifier subsumes another
        @param other - the classifier to check against
        @param theta_sub - See parameters above
        @param e0 - See parameters above
    """
    def _does_subsume(self, other, theta_sub, e0):
        return self.action == other.action and self._could_subsume(theta_sub, e0) and self._is_more_general(other)

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        if other == None:
            return False
        return self.id == other.id

"""
   The main NXCS class
"""
class nxcs:
    """
        Initializes an instance of Natural XCS
        @param parameters - A parameters instance (See above), containing the parameters for this system
        @param state_function - A function which returns the current state of the system, as a string
        @param reward_function - A function which takes a state and an action, performs the action and returns the reward
        @param eop_function - A function which returns whether the state is at the end of the problem
    """
    def __init__(self, parameters, state_function, reward_function, eop_function):
        self.pre_rho = 0
        self.parameters = parameters
        self.state_function = state_function
        self.reward_function = reward_function
        self.eop_function = eop_function
        self.population = []
        self.time_stamp = 0

        self.previous_match_set = None
        self.previous_action = None
        self.previous_reward = None
        self.previous_state = None

    """
        Prints the current population to stdout
    """
    def print_population(self):
        for i in sorted(self.population, key=lambda x: x.theta):
            print(i)

    """
       Classifies the given state, returning the class
       @param state - the state to classify
    """
    def classify(self, state):
        assert(len(state) == self.parameters.state_length), "Invalid state to classify"
        match_set = [classifier for classifier in self.population if _state_matches(classifier.condition, state)]
        predictions = self._generate_predictions(match_set)
        action = self._select_action(predictions)
        return action

    """
        Runs a single iteration of the learning algorithm for this NXCS instance
    """
    def run_experiment(self):
        curr_state = self.state_function()

        if self.eop_function(curr_state):
            #action = numpy.random.randint(self.parameters.num_actions)
            set_a = self._update_set(self.previous_state, curr_state, self.previous_action, self.previous_reward)
            _run_ga(set_a,self.previous_state)
        else:
            match_set = self._generate_match_set(curr_state)
            predictions = self._generate_predictions(match_set)
            action = self._select_action(predictions) #TODO: accroding to Pareto count

        if self.previous_state is not None and not self.eop_function(self.previous_state):
            set_a = self._update_set(self.previous_state, curr_state, self.previous_action, self.previous_reward)
            _run_ga(set_a,self.previous_state)

        self.previous_reward = self.reward_function(curr_state, action)
        self.previous_action = action
        self.previous_state = curr_state
        self.time_stamp = self.time_stamp + 1

    """
        Generates the match set for the given state, covering as necessary
        @param state - the state to generate a match set for
    """
    def _generate_match_set(self, state):
        assert(state is not None and len(state) == self.parameters.state_length), "Invalid state to generate match set from"
        set_m = []
        while len(set_m) == 0:
            set_m = [clas for clas in self.population if _state_matches(clas.condition, state)]
            if len(set_m) < self.parameters.theta_mna:#Cover
                clas = self._generate_covering_classifier(state, set_m)
                self._insert_to_population(clas)
                self._delete_from_population()
                set_m = []
        assert(len(set_m) >= self.parameters.theta_mna)
        return set_m

    """
        Deletes a classifier from the population, if necessary
    """
    def _delete_from_population(self):
        numerosity_sum = sum(clas.numerosity for clas in self.population)
        if numerosity_sum <= self.parameters.N:
            return

        average_fitness = sum(clas.fitness for clas in self.population) / numerosity_sum
        votes = [clas._delete_vote(average_fitness, self.parameters.theta_del, self.parameters.delta) for clas in self.population]
        vote_sum = sum(votes)
        choice = numpy.random.choice(self.population, p=[vote / vote_sum for vote in votes])
        if choice.numerosity > 1:
            choice.numerosity = choice.numerosity - 1
        else:
            self.population.remove(choice)

    """
        Inserts the given classifier into the population, if it isn't able to be
        subsumed by some other classifier in the population
        @param clas - the classifier to insert
    """
    def _insert_to_population(self, clas):
        assert(clas is not None), "Cannot insert None classifier"
        same = [c for c in self.population if (c.action, c.condition) == (clas.action, clas.condition)]
        if same:
            same[0].numerosity = same[0].numerosity + 1
            return
        self.population.append(clas)

    """
        Generates a classifier that conforms to the given state, and has an unused action from
        the given match set
        @param state - The state to make the classifier conform to
        @param match_set - The set of current matches
    """
    def _generate_covering_classifier(self, state, match_set):
        assert(match_set is not None and state is not None), "Cannot generate classifier from None arguments"
        assert(len(state) == self.parameters.state_length), "State is of invalid length"
        clas = classifier(self.parameters, state)
        used_actions = [classi.action for classi in match_set]
        available_actions = list(set(range(self.parameters.num_actions)) - set(used_actions))
        clas.action = numpy.random.choice(available_actions)
        clas.time_stamp = self.time_stamp
        return clas

    """
        Generates a prediction array for the given match set
        @param match_set - The match set to generate predictions for
    """
    #TODO:generate pareto Q for 4 directions, and its counts
    def _generate_predictions(self, match_set):
        assert(match_set is not None and len(match_set) >= self.parameters.theta_mna), "Invalid parameters to generate predictions"
        PA = [[0] * self.parameters.num_actions,[0] * self.parameters.num_actions]
        for clas in match_set:
            PA[0][clas.action] += clas.theta

        exp_pred = [numpy.exp(PA[0][i]) for i in range(self.parameters.num_actions)]
        pred_sum = sum(exp_pred)
        output = [i / pred_sum for i in exp_pred]
        assert(abs(sum(output) - 1) <= 0.0000001), "Output is not normalized: {}".format(output)
        return output

    """
        Selects the action to run from the given prediction array. Takes into account exploration
        vs exploitation
        @param predictions - The prediction array to generate an action from
    """
    def _select_action(self, predictions):
        assert(predictions is not None), "Predictions cannot be None"
        assert(abs(sum(predictions) - 1) <= 0.0000001), "Output is not normalized: {}".format(output)#not sure is it nxcs.output
        return numpy.random.choice(list(range(self.parameters.num_actions)), p=predictions)

    def _value_function_estimation(self, match_set):
        assert(match_set is not None), "Match set cannot be None"
        normalized_prediction_array = self._generate_predictions(match_set)
        ret = 0
        for i in range(self.parameters.num_actions):
            action_set = [clas for clas in match_set if clas.action == i]
            if len(action_set) == 0:
                continue
            ret = ret + normalized_prediction_array[i] * numpy.average([clas.prediction for clas in action_set], weights=[clas.fitness for clas in action_set])
        return ret

    """
       Updates the given action set's prediction, error, average size and fitness using the given decayed performance
       @param action_set - The set to update
       @param P - The reward to use
    """
    def _update_set(self, previous_state, current_state, action, reward):
        assert(previous_state is not None and current_state is not None and action is not None and reward is not None), "Illegal call to update set"
        assert(len(previous_state) == len(current_state) == self.parameters.state_length), "States not the correct length"
        assert(0 <= action < self.parameters.num_actions), "Invalid action performed"
        previous_match_set = self._generate_match_set(previous_state)

        if self.eop_function(current_state):
            current_match_set = None
            P = reward
        else:
            current_match_set = self._generate_match_set(current_state)
            P = reward + self.parameters.gamma * self._value_function_estimation(current_match_set)
            #TODO:P = reward + self.parameters.gamma * MAX PA (current_match_set)
            #TODO:Caculate R=R+(r-R)/n

        delta_t = P - self._value_function_estimation(previous_match_set)
        #TODO:It doesnt need delta_t

        action_set = [clas for clas in previous_match_set if clas.action == action]
        set_numerosity = sum(clas.numerosity for clas in action_set)


        #TODO: as same；prediction=P；
        #TODO: ERROR:if non donminate larger, then small error OR if hypervolum better, then smaller error
        for clas in action_set:
            clas.experience = clas.experience + 1
            if clas.experience < 1. / self.parameters.beta:
                clas.average_size = clas.average_size + (set_numerosity - clas.numerosity) / clas.experience
                clas.error = clas.error + (abs(P - clas.prediction) - clas.error) / clas.experience
                clas.prediction = clas.prediction + (P - clas.prediction) / clas.experience
            else:
                clas.average_size = clas.average_size + (set_numerosity - clas.numerosity) * self.parameters.beta
                clas.error = clas.error + (abs(P - clas.prediction) - clas.error) * self.parameters.beta
                clas.prediction = clas.prediction + (P - clas.prediction) * self.parameters.beta

        #Update fitness
        #TODO: if we use bigger error to represent better hypervolum, then bigger e, k=1, f bigger
        kappa = {clas: 1 if clas.error < self.parameters.e0 else self.parameters.alpha * (clas.error / self.parameters.e0) ** -self.parameters.nu for clas in action_set}
        accuracy_sum = sum(kappa[clas] * clas.numerosity for clas in action_set)

        for clas in action_set:
            clas.fitness = clas.fitness + self.parameters.beta * (kappa[clas] * clas.numerosity / accuracy_sum - clas.fitness)

        #Update policy parameter
        #TODO: Dont need Update policy parameter
        predictions = self._generate_predictions(previous_match_set)
        for clas in previous_match_set:
            state_feature = 0 - predictions[clas.action]
            if clas.action == action:
                state_feature = 1 - predictions[clas.action]

            clas.theta = clas.theta + (1. / self.parameters.rho0) * delta_t * state_feature

        if self.parameters.do_action_set_subsumption:
            self._action_set_subsumption(action_set);

    """
        Does subsumption inside the action set, finding the most general classifier
        and merging things into it
        @param action_set - the set to perform subsumption on
    """
    def _action_set_subsumption(self, action_set):
        assert(action_set is not None), "Cannot run action set subsumption on None actionset"
        cl = None
        for clas in action_set:
            if clas._could_subsume(self.parameters.theta_sub, self.parameters.e0):
                if cl == None or clas._is_more_general(cl):
                    cl = clas

        if cl:
            for clas in action_set:
                if cl._is_more_general(clas):
                    cl.numerosity = cl.numerosity + clas.numerosity
                    action_set.remove(clas)
                    self.population.remove(clas)

    """
        Runs the genetic algorithm on the given set, generating two new classifers
        to be inserted into the population
        @param action_set - the action set to choose parents from
        @param state - The state mutate with
    """
    def _run_ga(self, action_set, state):
        assert(not action_set is None), "Cannot run GA on None set"
        if len(action_set) == 0:
            return

        if self.time_stamp - numpy.average([clas.time_stamp for clas in action_set], weights=[clas.numerosity for clas in action_set]) > self.parameters.theta_ga:
            for clas in action_set:
                clas.time_stamp = self.time_stamp

            fitness_sum = sum([clas.fitness for clas in action_set])

            probs = [clas.fitness / fitness_sum for clas in action_set]
            parent_1 = numpy.random.choice(action_set, p=probs)
            parent_2 = numpy.random.choice(action_set, p=probs)
            child_1 = deepcopy(parent_1)
            child_2 = deepcopy(parent_2)
            child_1.id = classifier.global_id
            child_2.id = classifier.global_id + 1
            classifier.global_id = classifier.global_id + 2
            child_1.numerosity = 1
            child_2.numerosity = 1
            child_1.experience = 0
            child_2.experience = 0

            if numpy.random.rand() < self.parameters.crossover_rate:
                _crossover(child_1, child_2)
                child_1.prediction = child_2.prediction = 1.00 * numpy.average([parent_1.prediction, parent_2.prediction])
                child_1.error      = child_2.error      = 0.25 * numpy.average([parent_1.error, parent_2.error])
                child_1.fitness    = child_2.fitness    = 0.10 * numpy.average([parent_1.fitness, parent_2.fitness])
                child_1.theta = child_2.theta = numpy.average([parent_1.theta, parent_2.theta])

            for child in [child_1, child_2]:
                child._mutate(state, self.parameters.mutation_rate, self.parameters.num_actions)
                if self.parameters.do_GA_subsumption == True:
                    if parent_1._does_subsume(child, self.parameters.theta_sub, self.parameters.e0):
                        parent_1.numerosity = parent_1.numerosity + 1
                    elif parent_2._does_subsume(child, self.parameters.theta_sub, self.parameters.e0):
                        parent_2.numerosity = parent_2.numerosity + 1
                    else:
                        self._insert_to_population(child)
                else:
                    self._insert_to_population(child)

                self._delete_from_population()

"""
    Returns whether the given state matches the given condition
    @param condition - The condition to match against
    @param state - The state to match against
"""
def _state_matches(condition, state):
    assert(condition is not None and state is not None), "Cannot check matching against None state or condition"
    assert(len(condition) == len(state)), "Condition is not the same length as the state"
    return all(c == '#' or c == s for c, s in zip(condition, state))

"""
    Cross's over the given children, modifying their conditions
    @param child_1 - The first child to crossover
    @param child_2 - The second child to crossover
"""
def _crossover(child_1, child_2):
    assert(child_1 is not None and child_2 is not None), "Cannot run crossover on None child"
    x = numpy.random.randint(0, len(child_1.condition))
    y = numpy.random.randint(0, len(child_1.condition))

    child_1_condition = list(child_1.condition)
    child_2_condition = list(child_2.condition)

    for i in range(x, y):
        child_1_condition[i], child_2_condition[i] = child_2_condition[i], child_1_condition[i]

    child_1.condition = ''.join(child_1_condition)
    child_2.condition = ''.join(child_2_condition)

