import sqlite3
import sqlite_vec
from sqlite_vec import serialize_float32
db = sqlite3.connect('vec_database.db')
db.enable_load_extension(True)
sqlite_vec.load(db)
db.enable_load_extension(False)
db.execute('create table vector(id INTEGER PRIMARY KEY AUTOINCREMENT,vector BLOB)')
db.close()