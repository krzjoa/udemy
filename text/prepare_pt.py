# -*- coding: utf-8 -*-

from wolne_lektury import PanTadeusz

pt = PanTadeusz("Pan Tadeusz", "../data/pan-tadeusz.txt").load(raw=False)


def preprocess(pan_tadeusz):

    # Dokończyć pipeline - lematyzacja i POS-tagging
    for chapter in pan_tadeusz:
        print chapter



preprocess(pt)