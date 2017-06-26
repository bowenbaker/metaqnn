class Test:

    def get_all_time_fns(self):
        return [getattr(self, method) for method in dir(self) if callable(getattr(self, method))
                                                                and method.find('test_') == 0]