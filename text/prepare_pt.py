# -*- coding: utf-8 -*-

from wolne_lektury import PanTadeusz

pt = PanTadeusz("Pan Tadeusz", "../data/pan-tadeusz.txt").load(raw=False)



def clean_chapters(chapters):
    cleaned = []
    for ch in chapters:
        cleaned.append(ch.splitlines()[9:])
    return cleaned




def preprocess(cleaned_chapters):
    pass



#print pt["Księga druga"]
clch = clean_chapters(pt.chapters.values())
