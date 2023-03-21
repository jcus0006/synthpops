from . import base as spb

class Accommodation(spb.LayerGroup):
    """
    A class for accommodations and methods to operate on each.

    Args:
        kwargs (dict): data dictionary of the accommodation
    """

    def __init__(self, accomid=None, accomtypeid=None, roomsize=None, **kwargs):
        """
        Class constructor for empty accommodation.

        Args:
            **accomid (int)          : accom id
            **accomtypeid (int)      : accomtype id
            **member_uids (np.array) : ids of accom members
        """
        # set up default accom values
        super().__init__(accomid=accomid, accomtypeid=accomtypeid, roomsize=roomsize, **kwargs)
        self.validate()

        return

    def validate(self):
        """
        Check that information supplied to make a accommodation is valid and update
        to the correct type if necessary.
        """
        super().validate(layer_str='accommodation')
        return