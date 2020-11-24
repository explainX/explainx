from imports import *


class Analytics(object):
    def __init__(self):
        self.user = ref
        self.dict = dict()
        self.container = deque()

    @staticmethod
    def finding_address():
        try:
            val = get_mac()
            return val

        except Exception as e :
            return None

    @staticmethod
    def finding_ip():
        try:
            val = socket.gethostbyname(socket.gethostname())
            return val
        except Exception as e:
            return None

    @staticmethod
    def finding_system():
        try:
            return platform.system()
        except Exception as e:
            return None
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
