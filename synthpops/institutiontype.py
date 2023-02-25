from . import base as spb

class InstitutionType(spb.LayerGroup):
    """
    A class for individual institutions and methods to operate on each.

    Args:
        kwargs (dict): data dictionary of the institution
    """

    def __init__(self, insttypeid=None, **kwargs):
        """
        Class constructor for empty institution.

        Args:
            **instypeid (int)             : institution type id
            **member_uids (np.array) : institutions in institution type
        """
        # set up default industry values
        super().__init__(insttypeid=insttypeid, **kwargs)
        self.validate()

        return

    def validate(self):
        """
        Check that information supplied to make an institution type is valid and update
        to the correct type if necessary.
        """
        super().validate(layer_str='institutiontype')
        return