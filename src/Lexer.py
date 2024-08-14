class TokenTypes(object):

    names = ['EOF', 'SEMICOLON', 'NUMBER', 'THICKARROW', 'ARROW', 'DOUBLEARROW', 'TYPETAG', 'COMMA', 'LRB', 'RRB', 'ERROR', 'SLASH',
             'LSB', 'RSB', 'CIRC']
    
    (EOF, SEMICOLON, NUMBER, THICKARROW,  ARROW, DOUBLEARROW, TYPETAG, COMMA, LRB, RRB, ERROR, SLASH, LSB, RSB, CIRC) = range(15)

class Token(object):
    def __init__(self, tktype, text, line):
        # type is a numeric token type from TokenTypes
        # text is the lexeme

        self.type = tktype
        self.text = text
        self.line = line

    def __str__(self):
        # Converts token to string.
        return "<'%s', %s>" % (self.text, TokenTypes.names[self.type])


class Lexer:
    def __init__(self, st):
        self.input = st  # input string
        self.p = 0  # index of current character within self.input
        self.lineNum = 1  # current line number
        self.charNum = 1  # current character number within the line

        # Initialize the current character (self.c)

        if len(self.input) != 0:
            self.c = self.input[self.p]
        else:
            self.c = TokenTypes.EOF

    def nextToken(self):

        while self.c != TokenTypes.EOF:
            if self.c in [' ', '\t', '\n', '\r']:
                self._consume()
            elif self.c == ';':
                self._consume()
                return Token(TokenTypes.SEMICOLON, ';', self.lineNum)
            elif self.c == ',':
                self._consume()
                return Token(TokenTypes.COMMA, ',', self.lineNum)
            elif self.c == '(':
                self._consume()
                return Token(TokenTypes.LRB, '(', self.lineNum)
            elif self.c == ')':
                self._consume()
                return Token(TokenTypes.RRB, ')', self.lineNum)
            elif self.c == '/':
                self._consume()
                return Token(TokenTypes.SLASH, '/', self.lineNum)
            elif self.c == '[':
                self._consume()
                return Token(TokenTypes.LSB, '[', self.lineNum)
            elif self.c == ']':
                self._consume()
                return Token(TokenTypes.RSB, ']', self.lineNum)
            elif self.c == '^':
                self._consume()
                return Token(TokenTypes.CIRC, '^', self.lineNum)
            elif self.c == '-':  # '->'' is an ARROW, '-' followed by anything else is invalid.
                self._consume()
                if self.c == '>':
                    self._consume()
                    return Token(TokenTypes.ARROW, '->', self.lineNum)
                else:
                    return self.error()
            elif self.c == '<':
                self._consume()
                if self.c == '-':  # '->'' is an ARROW, '-' followed by anything else is invalid.
                    self._consume()
                    if self.c == '>':
                        self._consume()
                        return Token(TokenTypes.DOUBLEARROW, '<->', self.lineNum)
                    else:
                        return self.error()
                else:
                    return self.error
            elif self.c == '=':  # '==>' is a DOUBLEARROW, '==' followed by anything else is
                # invalid.
                self._consume()
                if self.c == '=':
                    self._consume()
                    if self.c == '>':
                        self._consume()
                        return Token(TokenTypes.THICKARROW, '==>', self.lineNum)
                    else:
                        return self.error()
            elif self.c == '#':
                # Consume everything until the end-of-line.
                while self.c != TokenTypes.EOF and self.c != '\n':
                    self._consume()
            elif self.c.isdigit():

                # Consume all contiguous digits and turn them into a NUMBER.
                lexeme = ""
                while self.c != TokenTypes.EOF and self.c.isdigit():
                    lexeme += self.c
                    self._consume()
                return Token(TokenTypes.NUMBER, lexeme, self.lineNum)

            elif self.c.isalpha():
                # Consume all contiguous alpha, digits, or _ characters, then check to
                # see if we recognize it as a reserved word.
                lexeme = ""
                while self.c != TokenTypes.EOF and (self.c.isalpha()):
                    lexeme += self.c
                    self._consume()
                # if lexeme == 'configuration':
                # t = Token(TokenTypes.CONFIGURATION, lexeme)
                # elif lexeme == 'productions':
                # t = Token(TokenTypes.PRODUCTIONS, lexeme)
                # else:
                t = Token(TokenTypes.TYPETAG, lexeme, self.lineNum)
                return t
            else:

                # Every other character is invalid.
                return self.error()

        return Token(TokenTypes.EOF, "<EOF>", self.lineNum)

    def _consume(self):
        # """Advance to the next character of input, or EOF."""
        # Update line number and character number.
        if self.c in ['\n', '\r']:
            self.lineNum = self.lineNum + 1
            self.charNum = 1
        else:
            self.charNum = self.charNum + 1

        # To the next character.
        self.p += 1
        if self.p >= len(self.input):
            self.c = TokenTypes.EOF
        else:
            self.c = self.input[self.p]

    def error(self):
        print("ERROR: Unexpected '{}' at [l:{},c:{}]".format(self.input[self.p], self.lineNum, self.charNum))
        return Token(TokenTypes.ERROR, self.input[self.p], self.lineNum)
