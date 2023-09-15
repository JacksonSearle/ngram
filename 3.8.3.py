import random

class Ngram:
    def __init__(self, file_path, n):
        self.n = n
        self.ngram_counts = {}
        self.start_token = '<bos>'
        train_words, test_words = self.process_text(file_path)
        self.build_ngrams(train_words)
        self.test_perplexity(test_words[:10])

    def build_ngrams(self, words):
        for i in range(len(words) - self.n + 1):
            ngram = tuple(words[i:i+self.n])
            self.process_ngram(ngram)

    def process_ngram(self, ngram):
        context = ngram[:-1]
        next_word = ngram[-1]
        
        if context not in self.ngram_counts:
            self.ngram_counts[context] = {}
        
        if next_word not in self.ngram_counts[context]:
            self.ngram_counts[context][next_word] = 1
        else:
            self.ngram_counts[context][next_word] += 1

    def process_text(self, file_path):
        with open(file_path, 'r') as reader:
            text = reader.read()
        words = text.split()
        index = int(len(words)*.9)
        train_words = words[:index]
        test_words = words[index:]
        train_words = [self.start_token for _ in range(self.n-1)] + train_words
        return train_words, test_words

    def generate_token(self, context):
        if context not in self.ngram_counts:
            return None
        dictionary = self.ngram_counts[context]
        values = list(dictionary.values())
        index = random.choices(range(len(values)), values)[0]
        keys = list(dictionary.keys())
        token = keys[index]
        return token

    def generate_sequence(self, num_tokens):
        sequence = [self.start_token] * (self.n - 1)
        for _ in range(num_tokens):
            context = tuple(sequence[-(self.n - 1):])
            next_token = self.generate_token(context)
            if next_token is None:
                break
            sequence.append(next_token)
        text = ' '.join(sequence[(self.n - 1):])
        return text
    
    def find_probability(self, word, context):
        if context not in self.ngram_counts:
            return 0
        dictionary = self.ngram_counts[context]
        if word not in dictionary.keys():
            return 0
        return dictionary[word]
    
    def test_perplexity(self, test_words):
        perplexity = 1
        sequence = [self.start_token] * (self.n - 1)
        for word in test_words:
            # See how likely this model would have generated it
            context = tuple(sequence[-(self.n - 1):])
            probability = self.find_probability(word, context)
            perplexity *= probability
        exponent = -1/len(test_words)
        if perplexity == 0:
            print('Cannot compute perplexity due to division by 0')
            return
        perplexity = perplexity ** exponent
        print(perplexity)

n = 2

model = Ngram('moby_dick.txt', n=n)
print(model.generate_sequence(100))

print('-'*50)

model = Ngram('life_in_forest.txt', n=n)
print(model.generate_sequence(100))
