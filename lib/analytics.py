from imports import *


class Analytics(object):
    def __init__(self):
        self.user = ref
        self.dict = dict()
        self.container = deque()

    @staticmethod
    def finding_address():
        val = get_mac()
        return val

    @staticmethod
    def finding_ip():
        val = socket.gethostbyname(socket.gethostname())
        return val

    @staticmethod
    def finding_system():
        return platform.system()

    def __setitem__(self, key, val):
        self.dict[key] = val

    def __getitem__(self, key):
        return self.dict[key]

    def random_string_generator(self, n=10):
        random_str = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(n))
        return random_str

    def insert_data(self):
        random_str = self.random_string_generator(15)
        self.container.append(random_str)
        self.user.child(random_str).set(self.dict)

    def update_data(self):
        val = self.container.pop()
        result = self.user.child(val)
        result.update({
            'finish_time': str(datetime.datetime.now())
        })
