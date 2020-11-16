import matplotlib.pyplot as plt
try:
    import datajoint as dj
except Exception as e:
    print("Could not import datajoint: {}".format(e))
    pass

import sys

ip = "localhost"
dbname = 'vgatPAG2'    # Name of the database subfolder with data

def start_connection():
    """
    Starts up the datajoint database and signs in with user and passoword + returns the database name
    """
    try:
        if dj.config['database.user'] != "root":
            try:
                dj.config['database.host'] = ip
            except Exception as e:
                print("Could not connect to database: ", e)
                return None, None

            dj.config['database.user'] = 'root'
            dj.config['database.password'] = 'fede'
            dj.config['database.safemode'] = True
            dj.config['safemode']= True

            dj.config["enable_python_native_blobs"] = True

            dj.conn()

        schema = dj.schema(dbname)
    except  Exception as e:
        raise ValueError(f'Failed to start server, make sure youve launched docker-compose from M/mysql-server.\n{e}')
    return schema


def print_erd():
    schema = start_connection()
    # dj.ERD(schema).draw()
    dj.Diagram(schema).draw()
    plt.show()


def manual_insert_skip_duplicate(table, key):
	try:
		table.insert1(key)
		return True
	except Exception as e:
		if isinstance(e, dj.errors.DuplicateError):
			return False # Just a duplicate warning
		elif isinstance(e, dj.errors.IntegrityError):
			raise ValueError("Could not insert in table, likely missing a reference to foreign key in parent table!\n{}\n{}".format(table.describe(), e))
		else:
			raise ValueError(e)

if __name__ == "__main__":
    start_connection()
    print_erd()
