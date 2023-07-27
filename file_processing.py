import os
import pandas as pd

#File Traversal
def traverse_files(path):
    
    df = pd.DataFrame()
#This will iternate over the entire directory and sort it into an empty pandas dataframe. 
    for root, directories, files in os.walk(path):
        
        for file in files:
            file_name, file_extension = os.path.splitext(file)
            if file_extension == ".cat":
                #Read and sort cat data into df
                with open(root+'/'+file, 'r') as f:
                    for line in f:
                        file_name = os.path.splitext(file)[0]
                        row = line.split(' ')
                        df = df.append({
                            'Filename': root+'/'+file_name,
                            'Points': row[0],
                            'Left Eye x': row[1],
                            'Left Eye y': row[2],
                            'Right Eye x': row[3],
                            'Right Eye y': row[4],
                            'Mouth x': row[5],
                            'Mouth y': row[6],
                            'Left Ear-1 x': row[7],
                            'Left Ear-1 y': row[8],
                            'Left Ear-2 x': row[9],
                            'Left Ear-2 y': row[10],
                            'Left Ear-3 x': row[11],
                            'Left Ear-3 y': row[12],
                            'Right Ear-1 x': row[13],
                            'Right Ear-1 y': row[14],
                            'Right Ear-2 x': row[15],
                            'Right Ear-2 y': row[16],
                            'Right Ear-3 x': row[17],
                            'Right Ear-3 y': row[18]
                        }, ignore_index=True)
            else:
                continue
    print(root)
    print(directories)
    print(files)           
    return df #.to_csv('cat_df_new.py')


#Only Runs if running this module. Will not run if imported elsewhere. 
if __name__ == "__main__":
	print(traverse_files('archive'))
    

    