from Production import Symbol
from Tree import OpTypes, OpNode, NodeTypes


class Interpreter:

    def __init__(self, productions):
        self.productions = productions

    def genOps(self, p, root, side='', optype=None):

        ret = False
        if optype == OpTypes.SWAPNODE:
            p.codeList.append(OpNode(OpTypes.SWAPNODE, p.lhs.value, p.rhs.value))
        elif side == 'rhs':
            if root is not None:
                if root.right is not None:
                    ret = ret | self.genOps(p, root.right, side)
                if root.left is not None:
                    ret = ret | self.genOps(p, root.left, side)

                if root.type == NodeTypes.NODE and (root.attrib['label'] not in p.lhs_table) and (
                        root.attrib['label'] not in p.working_table):
                    # is a node and is not in lhs, add node
                    p.codeList.append(OpNode(OpTypes.ADDNODE, root.value, label=root.attrib['label']))
                    p.working_table[root.attrib['label']] = Symbol(root.type, root.value, root.attrib)

                elif root.type == NodeTypes.EDGE and (root.attrib['label'] not in p.lhs_table) and (
                        root.attrib['label'] not in p.working_table):
                    # is an edge and is not in lhs, add edge
                    p.codeList.append(OpNode(OpTypes.ADDEDGE, root.left.attrib['label'], root.right.attrib['label']))
                    attrib = {"op1": root.left.attrib['label'], "op2": root.right.attrib['label']}
                    p.working_table[root.attrib['label']] = Symbol(root.type, root.value, attrib)

                elif root.type == NodeTypes.DOUBLE_EDGE and (root.attrib['label'] not in p.lhs_table) and (
                        root.attrib['label'] not in p.working_table):
                    # is an edge and is not in lhs, add edge
                    p.codeList.append(OpNode(OpTypes.ADDEDGE, root.left.attrib['label'], root.right.attrib['label']))
                    attrib = {"op1": root.left.attrib['label'], "op2": root.right.attrib['label']}
                    p.working_table[root.attrib['label']] = Symbol(root.type, root.value, attrib)
                     
                    p.codeList.append(OpNode(OpTypes.ADDEDGE, root.right.attrib['label'], root.left.attrib['label']))
                    attrib = {"op1": root.right.attrib['label'], "op2": root.left.attrib['label']}
                    p.working_table[root.attrib['label']] = Symbol(root.type, root.value, attrib)
                
                elif root.type == NodeTypes.OPERATION and root.value == "/" and (
                        root.left.attrib['label'] in p.lhs_table):

                    # is a remove and node in lhs, remove node
                    p.codeList.append(OpNode(OpTypes.REMNODE, root.left.value, label=root.left.attrib['label']))

                elif root.type == NodeTypes.OPERATION and root.value == "^" and (
                        root.left.attrib['label'] in p.lhs_table and root.right.attrib['label'] in p.lhs_table):

                    p.codeList.append(OpNode(OpTypes.MERGENODE, root.left.attrib['label'], root.right.attrib['label'],
                                             label=root.left.attrib['label']))

        elif side == 'lhs':
            if root is not None:
                if root.right is not None:
                    ret = ret | self.genOps(p, root.right, side)
                if root.left is not None:
                    ret = ret | self.genOps(p, root.left, side)

                if root.type == NodeTypes.NODE and (root.attrib['label'] not in p.rhs_table):
                    # is a node and is not in rhs, remove node
                    p.codeList.append(OpNode(OpTypes.REMNODE, root.value, label=root.attrib['label']))

                elif root.type == NodeTypes.EDGE and (root.attrib['label'] not in p.rhs_table):
                    # is an edge and is not on rhs, remove edge
                    p.codeList.append(OpNode(OpTypes.REMEDGE, root.left.attrib['label'], root.right.attrib['label']))

        return ret

    def generate(self):
        if self.productions is not None:
            for k, p in self.productions.items():
                if p.isSwap():
                    self.genOps(p, p.rhs, optype=OpTypes.SWAPNODE)
                else:
                    self.genOps(p, p.rhs, 'rhs')
                    self.genOps(p, p.lhs, 'lhs')

        return self.productions
