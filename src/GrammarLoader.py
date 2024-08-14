from Analyzer import Analyzer
from Interpreter import Interpreter

class GrammarLoader:
    
    def __init__(self,filename=''):       
        self.filename=filename
        
    def load(self,filename=''):
        
        if self.filename=='' and filename!='':
            self.filename=filename
        string=self.read(self.filename)
        
        
        self.productions=Analyzer(string).analyze()
        #self.printProductions()
        self.productions=Interpreter(self.productions).generate()
        #self.productions=Interpreter(Analyzer(string).analyze()).generate()
        
        return self.productions
        
        
    def read(self,filename=None):
        
        string=None
        if self.filename !=': ':
            file = open(self.filename, "r") 
            string=file.read()
            file.close()
        else: 
            print("ERROR: No grammar file provided!")
        return string    
    
    def printProductions(self):
        for key, value in self.productions.items():
            print( key, value.root,'\n')
            value.printSymbolTable() 
            value.printCodeList()

    def printProduction(self,key):
        value = self.productions[key]
        print( key, value.root,'\n')
        value.printSymbolTable()
        value.printCodeList()