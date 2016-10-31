
import sqlite_wrapper as db_wrapper


def save_word_vector(word_text, word_vector):
    return db_wrapper.db_insert_word_vector(word_text, word_vector)

def fetch_all_word_vectors():
    return db_wrapper.db_fetch_all_word_vectors()
