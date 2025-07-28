import requests


class DataverseAPIHandler:
    def __init__(self, base_url: str, access_token: str):
        self.base_url = base_url
        self.access_token = access_token
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "OData-MaxVersion": "4.0",
            "OData-Version": "4.0",
            "If-None-Match": "null",
        }

    def get(self, endpoint: str, timeout=60, params: dict | None = None, **kwargs) -> list[dict]:
        """
        Send a GET request to the Dataverse API, handling pagination automatically.

        :param str endpoint:
            The API endpoint (e.g., 'contacts', 'accounts')
        :param int timeout:
            Request timeout in seconds. Default is 60 seconds.
        :param dict | None params:
            Query parameters to include in the request
        :param kwargs:
            Additional arguments to pass to requests.get()

        :returns list[dict]:
            A list of records retrieved from the Dataverse API
        :raises requests.RequestException:
            If the request fails, raises requests.RequestException
        """
        url = f"{self.base_url}/api/data/v9.2/{endpoint}"
        values = []
        session = requests.Session()
        try:
            while url:
                response = session.get(
                    url,
                    headers=self.headers,
                    params=params if url.endswith(endpoint) else None,
                    timeout=timeout,
                    **kwargs,
                )
                response.raise_for_status()
                dataverse_data = response.json()
                values.extend(dataverse_data.get("value", []))
                url = dataverse_data.get("@odata.nextLink")
                params = None  # Only send params on the first request
        finally:
            session.close()
        return values

    def post(self, endpoint: str, data: dict, timeout=60, **kwargs) -> str | None:
        """
        Send a POST request to the Dataverse API to create a new record.


        :param str endpoint:
            The API endpoint (e.g., 'contacts', 'accounts')
        :param dict data:
            The JSON data to send in the request body
        :param int timeout:
            Request timeout in seconds. Default is 60 seconds.
        :param kwargs:
            Additional arguments to pass to requests.post()

        :returns str | None:
            The created record data ID or None if creation failed
        :raises requests.RequestException:
            If the request fails, raises requests.RequestException
        :raises ValueError:
            If data is None, raises ValueError
            indicating that data must be provided for POST requests.
        """
        if data is None:
            raise ValueError("Data must be provided for POST requests")

        url = f"{self.base_url}/api/data/v9.2/{endpoint}"
        session = requests.Session()
        try:
            response = session.post(
                url,
                headers=self.headers,
                json=data,
                timeout=timeout,
                **kwargs,
            )
            response.raise_for_status()

            # 204 No Content with the record ID in the OData-EntityId header
            if response.status_code == 204:
                # Extract the created record ID from the OData-EntityId header
                entity_id_header = response.headers.get("OData-EntityId", "")
                if "(" in entity_id_header:
                    record_id = entity_id_header.split("(")[-1].split(")")[0]
                else:
                    record_id = None
                return record_id
            return None
        finally:
            session.close()

    def delete(self, endpoint: str, record_id: str, timeout=60, **kwargs) -> bool:
        """
        Send a DELETE request to the Dataverse API to delete a record.

        :param str endpoint:
            The API endpoint (e.g., 'contacts', 'accounts')
        :param str record_id:
            The GUID of the record to delete
        :param int timeout:
            Request timeout in seconds. Default is 60 seconds.
        :param kwargs:
            Additional arguments to pass to requests.delete()

        :returns bool:
            True if deletion was successful, False otherwise
        :raises requests.RequestException:
            If the request fails, raises requests.RequestException
        """
        url = f"{self.base_url}/api/data/v9.2/{endpoint}({record_id})"
        session = requests.Session()
        try:
            response = session.delete(
                url,
                headers=self.headers,
                timeout=timeout,
                **kwargs,
            )
            response.raise_for_status()

            # DELETE requests typically return 204 No Content on success
            return response.status_code == 204
        finally:
            session.close()
