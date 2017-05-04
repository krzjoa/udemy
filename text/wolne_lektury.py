# -*- coding: utf-8 -*-
import re
from collections import OrderedDict


class Chapter(object):

    def __init__(self, name, content):
        self.name = name
        self.content = content



class AbstractChapter(Chapter):

    def __init__(self, name, content, abstract):
        super(AbstractChapter, self).__init__(name, content)
        self.abstract = abstract


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

        return self

    def _prettify(self):
        pass


    def as_one_string(self):
        from operator import add
        return reduce(add, self.content)


class PanTadeusz(WolnaLektura):

    def __init__(self, title, path):
        super(PanTadeusz, self).__init__(title, path)
        self.chapters = OrderedDict()

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.chapters.values()[item]
        else:
            return self.chapters[item]

    def __iter__(self):
        return self.chapters.itervalues()


    def _prettify(self):
        # Usunięcie tytułu i autora
        self.content = self.content[5:]

        # Podział na rozdziały
        pattern = r"(Księga\s(pierwsza|druga|trzecia|czwarta|piąta|szósta|siódma|ósma|dziewiąta|dziesiąta|jedenasta|dwunasta))|Epilog"

        whole_text = self.as_one_string()

        result = re.finditer(pattern, whole_text, flags=re.DOTALL | re.UNICODE)

        ch_limits, ch_titles = [], []

        for r in result:
            ch_limits.append(r.start()-1)
            ch_titles.append(r.group())

        ch_limits = ch_limits + [-1]

        for idx in xrange(len(ch_titles)):
            start = ch_limits[idx]
            stop = ch_limits[idx+1]
            self.chapters[ch_titles[idx]] = PanTadeusz._get_chapter(self.as_one_string()[start:stop], ch_titles[idx])

    @staticmethod
    def _get_chapter(whole_content, title):
        splitted = whole_content.split("\n")
        return AbstractChapter(title, splitted[9:], splitted[7:8])


def load_txt(path):
    with open(path, "r") as txt:
        return txt.readlines()


if __name__ == "__main__":
    pt = PanTadeusz("Pan Tadeusz", "../data/pan-tadeusz.txt")
    pt.load(raw=False)
    print pt['Księga pierwsza'].content

