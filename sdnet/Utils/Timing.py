# -*- coding: utf-8 -*-
# -*- author: JeremySun -*-
# -*- dating: 20/8/28 -*-


from datetime import datetime

time_elapsed = {}
start_time = {}
end_time = {}


def timer_start(name):
    start_time[name] = datetime.now()


def timer_stop(name):
    end_time[name] = datetime.now()
    if not name in time_elapsed:
        time_elapsed[name] = end_time[name] - start_time[name]
    else:
        time_elapsed[name] += end_time[name] - start_time[name]


def timer_report():
    total = 0
    for name in time_elapsed:
        total += time_elapsed[name].total_seconds()
    print('')
    print('----------------Timer Report----------------------')
    # lambda parameter_list]: expression
    for name, value in sorted(time_elapsed.items(), key=lambda item: -item[1].total_seconds()):
        print('%S: used time %s, %f%%' % ('{:20}'.format(name), str(value).split('.')[0], value.total_seconds() / total * 100.0))
    print('----------------------------------------------')
    print('')

