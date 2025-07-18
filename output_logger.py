class OutputLogger:
    def __init__(self):
        self.filename = "analise_classificacao.txt"
        self.file = open(self.filename, 'w', encoding='utf-8')
        self.write(f"Análise de Classificação \n")
    
    def write(self, text):
        self.file.write(text)
        self.file.flush()
        print(text, end='')
    
    def section(self, title, level=1):
        if level == 1:
            self.write(f"\n{title}\n")
            self.write("-" * len(title) + "\n")
        elif level == 2:
            self.write(f"\n{title}\n")
            self.write("." * len(title) + "\n")
        else:
            self.write(f"\n{title}:\n")
    
    def subsection(self, title):
        self.write(f"\n{title}\n")
    
    def close(self):
        self.file.close()