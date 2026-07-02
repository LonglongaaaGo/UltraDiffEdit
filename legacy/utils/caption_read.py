


def read_lines_from_file(file_path):
    """
    Read all lines from a text file and return them as a list of strings.

    Args:
    file_path (str): The path to the text file.

    Returns:
    list: A list of strings, each representing a line from the file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            # 读取所有行到一个列表
            lines = file.readlines()
            # 去掉每行末尾的换行符
            lines = [line.strip() for line in lines]
        return lines
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []



if __name__ == '__main__':
    # 使用示例
    file_path = 'path_to_your_file.txt'
    lines = read_lines_from_file(file_path)
    print(lines)

