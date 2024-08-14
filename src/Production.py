from Tree import NodeTypes


class Symbol:
    def __init__(self, stype, value, attrib=None):
        self.type = stype
        self.value = value
        if attrib is None:
            self.attrib = {}
        else:
            self.attrib = attrib
    def __str__(self, level=0):
        if self.type is not None:
            return "<'%s'[%s], %s>" % (NodeTypes.names[self.type], self.value, self.attrib)
        else:
            return "None"


class Production:
    def __init__(self, name, root, line):
        self.name = name
        self.line = line
        self.root = root
        self.codeList = []

        self.rhs = None
        self.lhs = None

        self.rhs_table = {}
        self.lhs_table = {}
        self.working_table = {}

        if self.root is not None:
            self.rhs = self.root.right
            self.lhs = self.root.left

    def __str__(self, branch='t'):
        ret = ""
        if branch == 't':
            ret += self.root.__str__()
        elif branch == 'r':
            ret += self.root.left.__str__()
        elif branch == 'l':
            ret += self.root.right.__str__()
        return ret

    def printSymbolTable(self):
        print("\t lhs:")
        for key, value in self.lhs_table.items():
            print("\t\t", key, value)

        print("\t rhs:")
        for key, value in self.rhs_table.items():
            print("\t\t", key, value)

    def printCodeList(self):
        print("\n\t{} codelist:".format(self.name))
        for op in self.codeList:
            if op is not None:
                print('\t\t', op)

    def isSwap(self):
        ret = False
        if self.rhs is not None and self.lhs is not None:
            ret = bool(self.rhs.type == NodeTypes.NODE) and bool(self.lhs.type == NodeTypes.NODE)
        return ret
