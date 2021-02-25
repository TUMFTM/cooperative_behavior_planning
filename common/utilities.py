from importlib import import_module


def dynamic_import(abs_module_path, class_name):
    """
    Created by:
    Marius Gruber

    Modified by:
    Christian Knies

    Description:
    This method dynamically imports the target class specified by the input arguments.
    The code snippet was taken from following link:
    https://www.bnmetrics.com/blog/dynamic-import-in-python3
    :param abs_module_path (dotted): Absolute path string to module containing the target class.
    :param class_name: Name of the target class.
    :return: Target class as type object.
    """
    module_object = import_module(abs_module_path)

    target_class = getattr(module_object, class_name)

    return target_class


class UniqueIDBuffer:
    """
    Created by:
    Marius Gruber

    Modified by:
    Christian Knies

    Description:
    This class implements a buffer mechanism that provides unique
    integer ids. An in-use id can be released by calling the free_id method.
    It will then be available again as an unique id.
    """
    def __init__(self):
        self.content = [0]

        self.highest_value = 0

    def increase_buffer(self):
        # Add new highest value
        self.highest_value += 1
        # Add to buffer
        self.content.append(self.highest_value)

    def get_unique_id(self):
        """
        Returns an unique integer id. Previously returned values
        will not be considered again unless the free_id method was called
        with the value as argument.
        :return: Unique integer id
        """
        # Check if buffer is exhausted
        if len(self.content) == 0:
            self.increase_buffer()

        # Get available value from buffer
        id = self.content.pop()

        return id

    def free_id(self, id):
        """
        The given id is added to the internal value storage and is available again as an unique integer id.
        """

        self.content.append(id)
