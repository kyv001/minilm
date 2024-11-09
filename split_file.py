def split_file(file_path: str, part_size: int=1024 ** 3):
    with open(file_path, 'rb') as f:
        while True:
            part = f.read(part_size)
            if not part:
                break
            yield part

if __name__ == '__main__':
    import sys
    import tqdm
    if len(sys.argv) < 2:
        print('Usage: python split_file.py file_path [part_size]')
        exit(1)
    file_path = sys.argv[1]
    part_size = int(sys.argv[2]) if len(sys.argv) > 2 else 1024 ** 3
    i = 0
    for part in tqdm.tqdm(split_file(file_path, part_size)):
        with open(f'{file_path}.part{i}', 'wb') as f:
            f.write(part)
        i += 1
