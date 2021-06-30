import time


# 代码执行时间装饰器
def time_count(func):
    def wrapper(*args, **kwargs):
        begin = time.time()
        func(*args, **kwargs)
        end = time.time()

        cost = end - begin
        print('本次用时:{:.2f}s'.format(cost))

    return wrapper

def write_csv(results, file_name):
    # 将测试集标签写入csv
    import csv
    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label'])
        writer.writerows(results)
