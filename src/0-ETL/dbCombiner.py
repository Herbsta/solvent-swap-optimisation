import sqlite3
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_tables(conn: sqlite3.Connection) -> List[str]:
    """Get all table names from a database."""
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND NOT name LIKE 'sqlite_%';")
    tables = [table[0] for table in cursor.fetchall()]
    return tables

def list_tables_info(conn: sqlite3.Connection, db_path: str) -> None:
    """Print detailed information about tables in the database."""
    tables = get_tables(conn)
    logger.info(f"\nDatabase {db_path} contains these tables:")
    for table in tables:
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info([{table}])")
        columns = cursor.fetchall()
        logger.info(f"- Table: {table}")
        logger.info(f"  Columns: {', '.join(col[1] for col in columns)}")
        
        try:
            cursor.execute(f"SELECT COUNT(*) FROM [{table}]")
            count = cursor.fetchone()[0]
            logger.info(f"  Row count: {count}")
        except sqlite3.OperationalError as e:
            logger.error(f"  Error getting row count: {e}")

def copy_table(source_conn: sqlite3.Connection, 
               dest_conn: sqlite3.Connection, 
               table_name: str, 
               prefix: str = "") -> None:
    """
    Copy a table from source database to destination database with optional prefix.
    """
    cursor = source_conn.cursor()
    new_name = f"{prefix}{table_name}"
    
    try:
        # Get table schema and modify it
        cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name=?;", (table_name,))
        schema_row = cursor.fetchone()
        
        if not schema_row:
            logger.error(f"Could not find schema for table: {table_name}")
            return
            
        original_schema = schema_row[0]
        
        # Create the new schema with the new table name
        new_schema = original_schema.replace(
            f"CREATE TABLE [{table_name}]",
            f"CREATE TABLE [{new_name}]"
        ).replace(
            f'CREATE TABLE "{table_name}"',
            f'CREATE TABLE "{new_name}"'
        ).replace(
            f"CREATE TABLE '{table_name}'",
            f"CREATE TABLE '{new_name}'"
        ).replace(
            f"CREATE TABLE {table_name}",
            f"CREATE TABLE {new_name}"
        )
        
        logger.info(f"Creating table {new_name}")
        dest_conn.execute(new_schema)
        dest_conn.commit()
        
        # Get column names for the INSERT statement
        cursor.execute(f"PRAGMA table_info([{table_name}])")
        columns = [info[1] for info in cursor.fetchall()]
        columns_str = ', '.join(f'[{col}]' for col in columns)
        placeholders = ','.join(['?' for _ in columns])
        
        # Select and insert data in chunks to handle large tables
        logger.info(f"Copying data from {table_name} to {new_name}")
        cursor.execute(f"SELECT {columns_str} FROM [{table_name}]")
        
        batch_size = 1000
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            
            dest_conn.executemany(
                f"INSERT INTO [{new_name}] ({columns_str}) VALUES ({placeholders})",
                rows
            )
            dest_conn.commit()
            
        logger.info(f"Successfully copied table {table_name} to {new_name}")
            
    except sqlite3.Error as e:
        logger.error(f"SQLite error processing table {table_name}: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing table {table_name}: {str(e)}")
        raise

def combine_databases(source_dbs: List[str], 
                     dest_db: str, 
                     prefixes: Optional[Dict[str, str]] = None) -> None:
    """
    Combine multiple SQLite databases into one with optional prefixes for table names.
    
    Args:
        source_dbs: List of paths to source database files
        dest_db: Path to destination database file
        prefixes: Dictionary mapping database paths to prefixes
                 Example: {'db1.sqlite': 'Bao_'}
    """
    prefixes = prefixes or {}
    
    logger.info(f"Creating destination database: {dest_db}")
    dest_conn = sqlite3.connect(dest_db)
    dest_conn.execute("PRAGMA foreign_keys = OFF;")
    dest_conn.execute("PRAGMA journal_mode = WAL;")
    
    try:
        for db_path in source_dbs:
            logger.info(f"\nProcessing database: {db_path}")
            try:
                source_conn = sqlite3.connect(db_path)
                source_conn.execute("PRAGMA foreign_keys = OFF;")
                
                # List all tables in the source database
                list_tables_info(source_conn, db_path)
                
                tables = get_tables(source_conn)
                prefix = prefixes.get(str(db_path), "")
                
                if not tables:
                    logger.warning(f"No tables found in {db_path}")
                    continue
                
                for table in tables:
                    try:
                        copy_table(source_conn, dest_conn, table, prefix)
                    except Exception as e:
                        logger.error(f"Failed to copy table {table}: {str(e)}")
                        continue
                
                source_conn.close()
                
            except sqlite3.Error as e:
                logger.error(f"Error processing database {db_path}: {str(e)}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error with database {db_path}: {str(e)}")
                continue
                
    finally:
        dest_conn.commit()
        dest_conn.execute("PRAGMA journal_mode = DELETE;")
        dest_conn.close()

def main():
    combine_databases(
        ['db/apiSolubilityDatabase.db', 'db/BaoSolubilityDatabase.db', 'db/pubchemSolubilityDatabase.db'],
        'db/MasterDatabase.db',
        {
            'db/apiSolubilityDatabase.db': 'api_',
            'db/BaoSolubilityDatabase.db': 'bao_',
            'db/pubchemSolubilityDatabase.db': 'pubchem_'
        }
    )

if __name__ == "__main__":
    main()
