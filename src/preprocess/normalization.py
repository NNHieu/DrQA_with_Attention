from vncorenlp import VnCoreNLP

class VncorenlpTokenizer(object):
    __instance = None
    @staticmethod 
    def getInstance():
        """ Static access method. """
        if VncorenlpTokenizer.__instance == None:
            VncorenlpTokenizer()
        return VncorenlpTokenizer.__instance
    def __init__(self):
        """ Virtually private constructor. """
        if VncorenlpTokenizer.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            VncorenlpTokenizer.__instance = VnCoreNLP("models/vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')

def vcn_word_segment(text):
    # To perform word (and sentence) segmentation
    sentences = VncorenlpTokenizer.getInstance().tokenize(text) 
    return [" ".join(sentence) for sentence in sentences]
    