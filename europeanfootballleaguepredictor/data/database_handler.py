import sqlite3 
import pandas as pd 
from loguru import logger 
import sys

class DatabaseHandler:
    def __init__(self, league:str, database: str):
        self.league = league
        self.database = database
    
    def get_data(self, table_names: str or list[str]) -> list[pd.DataFrame]:
        dataframe_list = []
        connection = None

        if isinstance(table_names, str):
            table_names = [table_names]

        try:
            # Connect to the SQLite database
            connection = sqlite3.connect(self.database)
            cursor = connection.cursor()

            for table_name in table_names:
                # Check if the table exists
                cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table_name,))
                result = cursor.fetchone()

                if result:
                    # Fetch data from the specified table
                    query = f"SELECT * FROM {table_name};"
                    dataframe_list.append(pd.read_sql_query(query, connection))
                    logger.info(f"Data fetched for table: {table_name}")
                else:
                    logger.warning(f"Table '{table_name}' does not exist in the database. Skipping.")

            return dataframe_list
        
        except sqlite3.Error as e:
            logger.error(f"SQLite error: {e}")

        finally:
            # Close the database connection
            if connection:
                connection.close()       
    
    def save_dataframes(self, dataframes: list, table_names: list):
        if isinstance(table_names, str):
            table_names = [table_names]
        if isinstance(dataframes, pd.DataFrame):
            dataframes = [dataframes]

        if len(dataframes) != len(table_names):
            logger.error("Length of dataframe_list must be equal to the length of table_names.")
            sys.exit(1)

        try:
            # Connect to the SQLite database
            connection = sqlite3.connect(self.database)
            cursor = connection.cursor()

            for df, table_name in zip(dataframes, table_names):
                # Save the DataFrame to the corresponding table in the database
                df.to_sql(table_name, connection, index=False, if_exists='replace')
                logger.info(f'Table {table_name} created/updated for {self.league} league.')
            
            logger.success('All tables created successfully.')

        except sqlite3.Error as e:
            print(f"SQLite error: {e}")

        finally:
            connection.commit()
            # Close the database connection
            if connection:
                connection.close()     