# -*- coding: utf-8 -*-
import re

class WolnaLektura(object):

    def __init__(self, title, path):
        self.path = path
        self.title = title
        self.author = False
        self.content = None

    def load(self, no_licence=True, raw=True):
        self.content = load_txt(self.path)

        # Ostatnie 15 wierszy
        if no_licence:
            self.content = self.content[:-15]

        if not raw:
            self._prettify()


    def _prettify(self):
        pass


    def as_one_string(self):
        from operator import add
        return reduce(add, self.content)


class PanTadeusz(WolnaLektura):

    def __init__(self, title, path):
        super(PanTadeusz, self).__init__(title, path)
        self.chapters = {}

    def __getitem__(self, item):
        return self.chapters[item]

    def _prettify(self):
        # Usunięcie tytułu i autora
        self.content = self.content[5:]

        # Podział na rozdziały
        pattern = r"(Księga\s(pierwsza|druga|trzecia|czwarta|piąta|szósta|siódma|ósma|dziewiąta|dziesiąta|jedenasta|dwunasta))|Epilog"

        whole_text = self.as_one_string()

        result = re.finditer(pattern, whole_text, flags=re.DOTALL | re.UNICODE)

        print [r.group() for r in result]

        # chapters_limits = [0] +[ch.start()-1 for ch in result] +[-1]
        # chapters_titles = [ch.groups() for ch in result]
        #
        # for r in result:
        #     print r.group(), r.start()
        # for idx in xrange(len(chapters_limits)-1):
        #     start = chapters_limits[idx]
        #     stop = chapters_limits[idx+1]
        #     print idx
        #     self.chapters[chapters_titles[idx]] = self.content[start:stop]


def load_txt(path):
    with open(path, "r") as txt:
        return txt.readlines()



if __name__ == "__main__":
    pt = PanTadeusz("Pan Tadeusz", "../data/pan-tadeusz.txt")
    pt.load(raw=False)
