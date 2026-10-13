"""Data structures for representing Fabric lakehouse paths."""

from dataclasses import dataclass
from pathlib import Path

from binaryrain_fabric_local_pyspark.config import CACHE_DIR, ONELAKE_HOST


@dataclass(frozen=True, slots=True)
class FabricPath:
    """Represent a location inside a Fabric lakehouse.

    :param str workspace:
        Workspace ID or name, which identifies the abfss file-system container.
    :param str lakehouse:
        Lakehouse ID or ``<name>.Lakehouse``.
    :param str area:
        Either ``Files`` or ``Tables``.
    :param str relative:
        Path below the area, for example ``onPrem/mssql/Kunden.parquet`` or ``dbo/kunden``.
    :param str host:
        OneLake host name. Default is the global OneLake endpoint.
    """

    workspace: str
    lakehouse: str
    area: str
    relative: str
    host: str = ONELAKE_HOST

    @property
    def lakehouse_path(self) -> str:
        """Return the path below the workspace container, beginning with the lakehouse."""
        return f"{self.lakehouse}/{self.area}/{self.relative}"

    @property
    def uri(self) -> str:
        """Return the full abfss URI for this lakehouse location."""
        return f"abfss://{self.workspace}@{self.host}/{self.lakehouse_path}"

    @property
    def local_path(self) -> Path:
        """Return the local cache path corresponding to this lakehouse location."""
        return CACHE_DIR / self.workspace / self.lakehouse / self.area / self.relative
