class Score:

    def __init__(self):
        # architecture/topology related
        self.number_of_segments = 0
        self.segment_score = 0
        self.segment_violations = 0
        self.segment_ratio = 0

        self.number_of_modules = 0
        self.moduleScore = 0

        self.costNormalized = 0
        self.totalCost = 0

        # network related
        self.connectivityScore = 0
        self.connectivityCount = 0

        self.loadScore = 0.0
        self.overloadCount = 0.0

        self.meanHops = 0.0

        self.latencyScore = 0.0

        self.score = 0

    def copy_score(self, score):
        self.connectivityScore = score.connectivityScore
        self.connectivityCount = score.connectivityCount

        self.number_of_segments = score.number_of_segments
        self.segment_score = score.segment_score
        self.segment_ratio = score.segment_ratio
        self.segment_violations = score.segment_violations

        self.number_of_modules = score.number_of_modules
        self.moduleScore = score.moduleScore

        self.costNormalized = score.costNormalized
        self.totalCost = score.totalCost

        self.meanHops = score.meanHops

        self.latencyScore = score.latencyScore

        self.loadScore = score.loadScore
        self.overloadCount = score.overloadCount

        self.score = score.score

        return self

    def __str__(self):
        sc_str = "{:.4} ->".format(self.score)
        sc_str += "[M:" + "{:.2}/{}]".format(self.moduleScore, self.number_of_modules)
        sc_str += "[S:" + "{:.2}/{}/{}]".format(self.segment_score, self.number_of_segments, self.segment_violations)

        sc_str += "[NR:" + "{:.3}/{}]".format(self.connectivityScore, int(self.connectivityCount))
        sc_str += "[NL:" + "{:.3}/{}]".format(self.loadScore, int(self.overloadCount))

        sc_str += "[LS: {:.2}/{:.2}/{:.2}]".format(self.latencyScore, self.loadScore, self.meanHops)
        sc_str += "[C:" + "{:.2}/{}]".format(self.costNormalized, self.totalCost)

        return sc_str

    def get_score(self) -> float:
        return self.score

    def get_string(self):
        return self.__str__()
