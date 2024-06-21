import React, { useState, useEffect } from 'react'; // Imports React, useState and useEffect hooks
import { View, Text, Button } from 'react-native'; // Imports necessary UI components from React Native

const API_URL = 'https://your-api-url.com/data'; // Replace with your actual API URL (String constant for API endpoint)

const App = () => {
  const [data, setData] = useState(null); // State variable to hold fetched data (initially null)
  const [isLoading, setIsLoading] = useState(false); // State variable for loading indicator (initially false)
  const [error, setError] = useState(null); // State variable to store any error message (initially null)

  const fetchData = async () => { // Asynchronous function to fetch data
    setIsLoading(true); // Set loading indicator to true before request
    setError(null); // Clear any previous error message

    try {
      const response = await fetch(API_URL); // Make a GET request to the API URL using fetch
      if (!response.ok) { // Check if the response was successful
        throw new Error(`API request failed with status ${response.status}`); // Throw error if not successful
      }
      const jsonData = await response.json(); // Parse the response as JSON
      setData(jsonData); // Update state with the parsed data
    } catch (error) {
      setError(error.message); // Set error state with the error message
    } finally {
      setIsLoading(false); // Set loading indicator back to false regardless of outcome
    }
  };

  useEffect(() => {
    fetchData(); // Fetch data on component mount (empty dependency array)
  }, []);

  return (
    <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
      {isLoading && <Text>Loading data...</Text>} {/* Conditionally render loading text */}
      {error && <Text style={{ color: 'red' }}>{error}</Text>} {/* Conditionally render error message */}
      {data && (
        <Text>API data: {JSON.stringify(data, null, 2)}</Text> // Conditionally render fetched data (formatted)
      )}
      <Button title="Fetch Data" onPress={fetchData} /> {/* Button to trigger data fetching */}
    </View>
  );
};

export default App;