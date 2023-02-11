from . import base as spb

class Industry(spb.LayerGroup):
    """
    A class for individual workplaces and methods to operate on each.

    Args:
        kwargs (dict): data dictionary of the workplace
    """

    def __init__(self, indid=None, **kwargs):
        """
        Class constructor for empty industry.

        Args:
            **indid (int)             : indid id
            **member_uids (np.array) : workplaces in industry
        """
        # set up default industry values
        super().__init__(indid=indid, **kwargs)
        self.validate()

        return

    def validate(self):
        """
        Check that information supplied to make a industry is valid and update
        to the correct type if necessary.
        """
        super().validate(layer_str='industry')
        return