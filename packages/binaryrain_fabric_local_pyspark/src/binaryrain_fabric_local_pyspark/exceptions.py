"""Exceptions raised by the package.

Every package specific error derives from :class:`FabricLocalSparkError`, so callers can
catch one type. Configuration problems also derive from ``ValueError`` to stay compatible
with code that already catches it.
"""


class FabricLocalSparkError(Exception):
    """Base class for all errors raised by this package."""


class FabricConfigurationError(FabricLocalSparkError, ValueError):
    """Raised when a required environment variable or setting is missing."""


class FabricAuthenticationError(FabricLocalSparkError):
    """Raised when no Azure access token can be obtained."""


class FabricDownloadError(FabricLocalSparkError):
    """Raised when data cannot be downloaded from OneLake."""


class FabricCacheError(FabricLocalSparkError):
    """Raised when the local cache cannot be written or replaced."""


class FabricQueryError(FabricLocalSparkError):
    """Raised when a query against the Fabric SQL endpoint fails."""


class SparkLaunchError(FabricLocalSparkError):
    """Raised when the local Spark JVM or session cannot be started."""
