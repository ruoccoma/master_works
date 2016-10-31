
import sqlite_wrapper as db_wrapper


def fetch_all_word_vectors():
	return db_wrapper.db_fetch_all_word_vectors()


def fetch_word_vector(word, default_return=None):
	return db_wrapper.db_fetch_word_vector(word, default=None)
