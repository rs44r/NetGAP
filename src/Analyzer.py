from Lexer import Lexer, TokenTypes
from Production import Production, Symbol
from Tree import TreeNode, NodeTypes


def parseError(token, count):
    print("ERROR [r:{} l:{}]: Unexpected '{}'".format("r" + str(count), token.line, token.text))


class Analyzer:
    def __init__(self, string):
        self.lexer = Lexer(string)

    def trimCheck(self, production, root):

        ret = False
        if root is not None:
            if root.right is not None:
                ret = ret | self.trimCheck(production, root.right)
            if root.left is not None:
                ret = ret | self.trimCheck(production, root.left)

            if root.type == NodeTypes.EDGE or root.type ==  NodeTypes.DOUBLE_EDGE:
                if root.right.type == NodeTypes.EDGE or root.right.type == NodeTypes.LIST:

                    root.type = NodeTypes.LIST
                    root.value = None

                    new = TreeNode(NodeTypes.EDGE, None)
                    new.left = root.left
                    newright = root.right

                    while newright.type == NodeTypes.EDGE or newright.type == NodeTypes.LIST:
                        newright = newright.left

                    new.right = newright
                    root.left = new

                    new.value = str(new.left.value) + '->' + str(new.right.value)
                    new.attrib['label'] = str(new.left.attrib['label']) + '->' + str(new.right.attrib['label'])

                else:
                    root.value = str(root.left.value) + '->' + str(root.right.value)
                    if root.attrib is None:
                        root.attrib = {}
                    root.attrib['label'] = str(root.left.attrib['label']) + '->' + str(root.right.attrib['label'])

            if root.type == NodeTypes.OPERATION:
                if root.left.type != NodeTypes.NODE:
                    print("ERROR [r:{} l:{}]: Invalid Operation: '{}', all operands of should be NODES".format(
                        production.name, production.line, root.value))
                    ret = True

        return ret

    def initSymbolTable(self, production):
        if production is not None:
            self.fillSymbols(production, production.rhs, production.rhs_table)
            self.fillSymbols(production, production.lhs, production.lhs_table)

    def fillSymbols(self, production, root, table):

        if root is not None:
            self.fillSymbols(production, root.left, table)
            self.fillSymbols(production, root.right, table)

            if root.type == NodeTypes.NODE:
                key = root.attrib['label']

                if key not in table:
                    table[key] = Symbol(root.type, root.value, root.attrib)
                else:
                    if table[key].attrib != root.attrib:
                        print("ERROR [r:{} l:{}]: Conflicting definitions for '{}'".format(production.name,
                                                                                           production.line, key))
                    else:
                        for k, v in table[key].attrib.items():
                            if v != root.attrib[k]:
                                print("ERROR [r:{} l:{}]: Conflicting definitions for '{}'".format(production.name,
                                                                                                   production.line,
                                                                                                   key))

            if root.type == NodeTypes.EDGE:

                key = root.left.attrib['label'] + "->" + root.right.attrib['label']
                attrib = {"op1": root.left.attrib['label'], "op2": root.right.attrib['label']}

                if key not in table:
                    table[key] = Symbol(root.type, root.value, attrib)
                else:
                    if table[key].attrib != attrib:
                        print("ERROR [r:{} l:{}]: Conflicting definitions for '{}'".format(production.name,
                                                                                           production.line, key))
                    else:
                        for k, v in table[key].attrib.items():
                            if v != attrib[k]:
                                print("ERROR [r:{} l:{}]: Conflicting definitions for '{}'".format(production.name,
                                                                                                   production.line,production.name))

    def parse(self):

        productions = {}

        lhs = None
        rhs = None

        current = TreeNode(None, None)
        root = current
        nexttk = self.lexer.nextToken()
        count = 0

        while True:

            if nexttk.type == TokenTypes.TYPETAG:
                ident = nexttk.text
                nexttk = self.lexer.nextToken()

                while True:
                    if nexttk.type == TokenTypes.TYPETAG:
                        ident += nexttk.text
                        nexttk = self.lexer.nextToken()
                    else:
                        break
                current.value = ident
                current.attrib = {'label': ident}
                current.type = NodeTypes.NODE

            elif nexttk.type == TokenTypes.SEMICOLON:

                if rhs is None:
                    lhs = None

                if root.type is None:
                    parseError(nexttk, count)
                    break

                rhs = root

                new = TreeNode(NodeTypes.PRODUCTION, None)
                root = new
                root.left = lhs
                root.right = rhs

                productions['r' + str(count)] = Production('r' + str(count), root, nexttk.line)
                count += 1
                lhs = TreeNode(None, None)
                rhs = None
                root = lhs
                current = root

                nexttk = self.lexer.nextToken()

            elif nexttk.type == TokenTypes.THICKARROW:
                if root.type is not None:
                    if lhs is not None and rhs is not None:
                        parseError(nexttk, count)
                        break
                    else:
                        lhs = root
                        current = TreeNode(None, None)
                        root = current
                        rhs = root
                        nexttk = self.lexer.nextToken()
                else:
                    parseError(nexttk, count)
                    break

            elif nexttk.type == TokenTypes.SLASH:

                current.type = NodeTypes.OPERATION
                current.value = nexttk.text
                current.attrib = None
                current.left = TreeNode(None, None)
                current = current.left
                nexttk = self.lexer.nextToken()

            elif nexttk.type == TokenTypes.CIRC:

                # self.parseError(nexttk,count )
                # break

                new = TreeNode(current.type, current.value, current.attrib)
                new.left = current.left
                new.right = current.right

                current.type = NodeTypes.OPERATION
                current.value = nexttk.text
                current.attrib = None
                current.left = new
                current.right = TreeNode(None, None)

                current = current.right
                nexttk = self.lexer.nextToken()


            elif nexttk.type == TokenTypes.ARROW:

                new = TreeNode(current.type, current.value, current.attrib)
                new.left = current.left
                new.right = current.right

                current.type = NodeTypes.EDGE
                current.value = None
                current.attrib = None
                current.left = new
                current.right = TreeNode(None, None)
                current = current.right
                nexttk = self.lexer.nextToken()
                
                

            elif nexttk.type == TokenTypes.DOUBLEARROW:


                new = TreeNode(current.type, current.value, current.attrib)
                new.left = current.left
                new.right = current.right

                current.type = NodeTypes.DOUBLE_EDGE
                current.value = None
                current.attrib = None
                current.left = new
                current.right = TreeNode(None, None)
                current = current.right
                nexttk = self.lexer.nextToken()

            elif nexttk.type == TokenTypes.COMMA:

                new = TreeNode(NodeTypes.LIST, None)
                new.left = root
                new.right = TreeNode(None, None)
                root = new
                current = new.right
                nexttk = self.lexer.nextToken()

            elif nexttk.type == TokenTypes.NUMBER:

                current.attrib = {'label': (str(current.value) + nexttk.text)}
                nexttk = self.lexer.nextToken()

            elif nexttk.type == TokenTypes.LSB:

                nexttk = self.lexer.nextToken()

                if nexttk.type == TokenTypes.NUMBER:

                    thismin = nexttk.text

                    nexttk = self.lexer.nextToken()
                    if nexttk.type == TokenTypes.COMMA:

                        nexttk = self.lexer.nextToken()
                        if nexttk.type == TokenTypes.NUMBER:

                            thismax = nexttk.text

                            nexttk = self.lexer.nextToken()

                            if nexttk.type != TokenTypes.RSB:
                                parseError(nexttk, count)
                                break
                        else:
                            parseError(nexttk, count)
                            break

                    elif nexttk.type == TokenTypes.RSB:

                        thismax = thismin
                        thismin = '0'
                    else:
                        parseError(nexttk, count)
                        break

                    nexttk = self.lexer.nextToken()

                    if current.attrib is None:
                        current.attrib = {'min': thismin, 'max': thismax}
                    else:
                        current.attrib['min'], current.attrib['max'] = thismin, thismax



                else:
                    parseError(nexttk, count)
                    break

            elif nexttk == TokenTypes.RSB:

                parseError(nexttk, count)
                break

            elif nexttk.type == TokenTypes.ERROR or nexttk.type == TokenTypes.EOF:
                break

            else:
                parseError(nexttk, count)
                # nexttk=self.lexer.nextToken()
                break

        self.productions = productions

    def syntax(self):
        self.parse()

    def semantics(self):

        for key in list(self.productions.keys()):
            value = self.productions[key]
            if self.trimCheck(value, value.root):
                print("ERROR [r:{} l:{}]: Semantic Error, rule {} will be removed".format(value.name, value.line,
                                                                                          value.name))
                del self.productions[key]
            else:
                self.initSymbolTable(value)

    def printProductions(self):
        for key, value in self.productions.items():
            print(key, value.root)
            value.printSymbolTable()

    def analyze(self):

        self.syntax()
        self.semantics()

        return self.productions
