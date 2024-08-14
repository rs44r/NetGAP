class NodeTypes:
    (NODE, NUMBER, PRODUCTION, EDGE, LIST, OPERATION,DOUBLE_EDGE) = range(7)

    names = ['NODE', 'NUMBER', 'PRODUCTION', 'EDGE', 'LIST', 'OPERATION','DOUBLE_EDGE']


class OpTypes:
    (ADDEDGE, ADDNODE, REMEDGE, REMNODE, MERGENODE, SWAPNODE) = range(6)

    names = ['ADDEDGE', 'ADDNODE', 'REMEDGE', 'REMNODE', 'MERGENODE', 'SWAPNODE']


class OpNode:
    def __init__(self, ntype, op1=None, op2=None, label=None):
        self.type = ntype
        self.op1 = op1
        self.op2 = op2
        self.label = label

    def __str__(self):
        ret = ''
        if self.type == OpTypes.ADDEDGE or self.type == OpTypes.REMEDGE or \
                self.type == OpTypes.SWAPNODE or self.type == OpTypes.MERGENODE:

            if self.label is not None:
                ret += '<{}:{},{}> as {}'.format(OpTypes.names[self.type], self.op1, self.op2, self.label)
            else:
                ret += '<{}:{},{}>'.format(OpTypes.names[self.type], self.op1, self.op2)
        elif self.type == OpTypes.ADDNODE or self.type == OpTypes.REMNODE:
            if self.label is not None:
                ret += '<{}:{}> as {}'.format(OpTypes.names[self.type], self.op1, self.label)
            else:
                ret += '<{}:{}>'.format(OpTypes.names[self.type], self.op1)
        return ret


class TreeNode:

    def __init__(self, ntype, value, attrib=None):
        self.type = ntype
        self.value = value
        if attrib is None:
            self.attrib = {}
        else:
            self.attrib = attrib
        self.right = None
        self.left = None

    def __str__(self, level=0):
        if self.type is None:
            ret = "\t" * level + "<EMPTY>\n"
        else:
            if self.value is not None:
                if self.type == NodeTypes.NODE or self.type == NodeTypes.EDGE:
                    ret = "\t" * level + "<'%s'[%s], %s>" % (NodeTypes.names[self.type], self.value, self.attrib) + "\n"
                else:
                    ret = "\t" * level + "<'%s', %s>" % (NodeTypes.names[self.type], self.value) + "\n"

            else:
                ret = "\t" * level + "<%s>" % (NodeTypes.names[self.type]) + "\n"

        if (self.left is not None) or (self.right is not None):

            if self.left is not None:
                ret += self.left.__str__(level + 1)
            else:
                ret += "\t" * (level + 1) + "<EMPTY>\n"

            if self.right is not None:
                ret += self.right.__str__(level + 1)
            else:
                ret += "\t" * (level + 1) + "<EMPTY>\n"

        return ret
