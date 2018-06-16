lines = tuple(open('lyrics_with_metadata.txt', 'r'))
with open('lyrics.txt', 'a') as file:
    for line in lines:
        if line.startswith("[") == False:
            file.write(line)
