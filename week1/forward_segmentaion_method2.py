# 分词方法最大正向切分的第二种方法
import time


# 加载词前缀字典
# 用0和1来区分是前缀还是真词
# 需要注意有的词前缀也是真词，在记录时不要互相覆盖
def load_prefix_word_dict(path):
    prefix_dict = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            word = line.split()[0]
            for i in range(1, len(word)):
                if word[:i] not in prefix_dict:
                    prefix_dict[word[:i]] = 0
            prefix_dict[word] = 1
    return prefix_dict


# 输入字符串和字典，返回词的列表
def cut_method2(string, prefix_dict):
    if string == '':
        return []
    words = []
    start_index, end_index = 0, 1
    window = string[start_index:end_index]
    first_word = window
    while start_index < len(string):
        # 窗口没有在词典里出现
        if window not in prefix_dict or end_index > len(string):
            words.append(first_word)
            start_index += len(first_word)
            end_index = start_index + 1
            window = string[start_index:end_index]
            first_word = window

        # 窗口为词
        elif prefix_dict[window] == 1:
            first_word = window
            end_index += 1
            window = string[start_index:end_index]
        # 窗口为词前缀
        elif prefix_dict[window] == 0:
            end_index += 1
            window = string[start_index:end_index]
    return words


# cut_method是切割函数
# output_path是输出路径
def main(word_dict_path, cut_method, input_path, output_path):
    word_dict = load_prefix_word_dict(word_dict_path)
    writer = open(output_path, "w", encoding="utf-8")
    start_time = time.time()
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            words = cut_method(line, word_dict)
            writer.write(" / ".join(words) + "\n")
    writer.close()
    print("耗时：", time.time() - start_time)
    return


if __name__ == "__main__":
    main("dict.txt", cut_method2, "corpus.txt", "cut_method2_output2.txt")






















