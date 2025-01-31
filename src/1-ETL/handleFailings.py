import sqlite3

def copy_failed_tables():
    source_db = 'db/MasterDatabase.db'
    destination_db = 'helper/failedListings.db'

    # Connect to the source database
    source_conn = sqlite3.connect(source_db)
    source_cursor = source_conn.cursor()

    # Connect to the destination database
    dest_conn = sqlite3.connect(destination_db)
    dest_cursor = dest_conn.cursor()

    # Get the list of tables that start with 'failed_'
    source_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'failed_%'")
    failed_tables = source_cursor.fetchall()

    for table in failed_tables:
        table_name = table[0]
        # Create the table in the destination database
        source_cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        create_table_sql = source_cursor.fetchone()[0]
        dest_cursor.execute(create_table_sql)

        # Copy the data from the source table to the destination table
        source_cursor.execute(f"SELECT * FROM {table_name}")
        rows = source_cursor.fetchall()
        for row in rows:
            placeholders = ', '.join(['?'] * len(row))
            dest_cursor.execute(f"INSERT INTO {table_name} VALUES ({placeholders})", row)

    # Commit changes and close connections
    dest_conn.commit()
    source_conn.close()
    dest_conn.close()

# Call the function to copy the failed tables
copy_failed_tables()