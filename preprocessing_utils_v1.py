from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from operator import itemgetter

tokenizer = RegexpTokenizer(r'\w+')  # punctuation removal and tokenization
ps = PorterStemmer()  # reducing or chopping the words into their root forms

corrections = {'ad': 'and', 'i': 'is', 's': 'is', 'ia': 'is a', 'adge': 'edge', 'al': 'at', 't': 'at',
               'tleast': 'at least', 'atleast': 'at least', 'ablue': 'a blue', 'ans': 'and', 'bkack': 'black',
               'bloxk': 'block', 'abox.': 'a box', 'bow': 'box', 'blicks': 'block', 'bo': 'box', 'boxes': 'box',
               'ble': 'blue',
               'blacks': 'black', 'blccks': 'black', 'back': 'black', 'bellow': 'below', 'contains': 'contain',
               'containing': 'conatin', 'opis': 'is', 'isa': 'is a', 'exacrly': 'exactly', 'exacts': 'exactly',
               'eactly': 'exactly', 'exacty': 'exactly', 'cirlce': 'circle', 'ciircles': 'circles',
               'cirlce': 'circle',
               'colour': 'color', 'colours': 'colors', 'coloured': 'color', 'colour.': 'color', 'colored': 'color',
               'closely': 'close', 'ha': 'have', 'having': 'have',
               'including': 'include', 'sqaures': 'squares', 'egde': 'edge',

               'leats': 'least', 'lest': 'least', 'lease': 'least', 'squere': 'square', 'squares': 'square',
               'touhing': 'touch', 'tocuhing': 'touch', 'traingle': 'triangle', 'traingles': 'triangle',
               'trianlge': 'triangle',
               'hte': 'the', 'thee': 'the', 'then,idle': 'the middle', 'theer': 'there', 'od': 'odd',
               'objetcs': 'objects',
               'tow': 'two', 'wirh': 'with', 'wwith': 'with', 'wth': 'with', 'wih': 'with',
               'yelow': 'yellow', 'yelloe': 'yellow', 'yelllow': 'yellow'
               }

corrections2 = {'above': 'top', 'blocks': 'block', 'items': 'item', 'objects': 'item', 'towers': 'tower',
                'triangles': 'triangle', 'colors': 'color', 'squares': 'square', 'attached': 'attach',
                'stack': 'stack', 'boxes': 'box', 'shapes': 'shape', 'numbers': 'number', 'corners': 'corner',
                'positions': 'position', 'bases': 'base', 'kinds': 'shape', 'below': 'under', 'grey': 'black',

                'object': 'item', 'it': 'item', 'underneath': 'under', 'roof': 'top', 'include': 'contain',
                'both': 'two',
                'they': 'item', 'objects': 'item', 'beneath': 'under', 'them': 'item', 'type': 'shape',
                'over': 'on',
                'line': 'edge', 'ones': 'item', 'stacked': 'stack', 'single': 'one', 'corners': 'corner',
                'attached': 'attach',
                'touching': 'touch', 'bottom': 'base', 'alternately': 'different', 'odd': 'different',
                'wall': 'edge',
                'smaller': 'small', 'lot': 'many', 'multiple': 'many', 'none': 'no', 'rectangle': 'square',
                # 'box':'square',
                'even': 'same', 'first': 'one', 'second': 'two', 'third': 'three', 'traingles': 'triangle',
                'circles': 'circle', 'block': 'square', 'total': 'all', 'side': 'edge',
                }
numbers = {'1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five', '6': 'six', '7': 'seven', '8': 'eight'}


def prepare_data(data, all=False):
    tuple_keys = ('sentence', 'structured_rep', 'label')
    get_keys = itemgetter(*tuple_keys)
    if all:
        data = [dict(zip(tuple_keys, get_keys(annotation))) for annotation in data]
    else:
        data = [dict(zip(tuple_keys, get_keys(annotation))) for annotation in data if
                annotation['label'] == 'true']
    return data


def preprocess(text):
    text = text.lower().strip()
    words = tokenizer.tokenize(text)  # removing punctuation and tokenizing
    words = [word if not word in numbers.keys() else numbers[word] for word in words]
    words = [word if not word in corrections.keys() else corrections[word] for word in words]
    words = [word if not word in corrections2.keys() else corrections2[word] for word in words]
    text = ' '.join(words)
    return text
