from collections import namedtuple
from typing import Union


class OutputMixIn:
    """
    MixIn to give namedtuple some access capabilities of a dictionary
    """

    def __getitem__(self, k):
        if isinstance(k, str):
            return getattr(self, k)
        else:
            return super().__getitem__(k)

    def get(self, k, default=None):
        return getattr(self, k, default)

    def items(self):
        return zip(self._fields, self)

    def keys(self):
        return self._fields

    def iget(self, idx: Union[int, slice]):
        """Select item(s) row-wise.

        Args:
            idx ([int, slice]): item to select

        Returns:
            Output of single item.
        """
        return self.__class__(*(x[idx] for x in self))


class TupleOutputMixIn:
    """MixIn to give output a namedtuple-like access capabilities with ``to_network_output() function``."""

    def to_network_output(self, **results):
        """
        Convert output into a named (and immuatable) tuple.

        This allows tracing the modules as graphs and prevents modifying the output.

        Returns:
            named tuple
        """
        if hasattr(self, "_output_class"):
            Output = self._output_class
        else:
            OutputTuple = namedtuple("output", results)

            class Output(OutputMixIn, OutputTuple):
                pass

            self._output_class = Output

        return self._output_class(**results)
