# -*- coding: utf-8 -*-

from wolne_lektury import PanTadeusz
from pyMorfologik import Morfologik
from pyMorfologik.parsing import ListParser

pt = PanTadeusz("Pan Tadeusz", "../data/pan-tadeusz.txt").load(raw=False)


def text_pipe(book):

    # Morfologik
    parser = ListParser()
    stemmer = Morfologik()

    processed_chapters = []

    for chapter in book:
        # Wyciąganie treści
        content = chapter.content

        # Lowercasing
        lowercased = [chunk.lower() for chunk in content]

        # Zlematyzowane
        stemmed = stemmer.stem(lowercased, parser)

        # Przekształcanie
        final = [tup[1].keys() for tup in stemmed]

        # Finałowo
        final = [tup[0] for tup in final if tup]

        processed_chapters.append(final)

    return processed_chapters


processed = text_pipe(pt)

print processed[0]


