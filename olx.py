import streamlit as st
import pickle
import numpy as np
import sklearn
st.title('Car Price Predictor Faraz')
df = pickle.load(open('OLX_data_frame.pkl', 'rb'))
pipe = pickle.load(open('pipe (2).pkl','rb'))
company=st.selectbox('Company Name',df['Make'].unique())
model=st.selectbox('Model',df['Model'].unique())
year=st.number_input("Model Year", 0, 4000)
km=st.number_input("KM's driven", 0, 100000000)
fuel=st.selectbox('Fuel Type',df['Fuel'].unique())
city=st.selectbox('Registration city',df['Registration city'].unique())
document=st.selectbox('Car documents',df['Car documents'].unique())
assembly=st.selectbox('Assembly',df['Assembly'].unique())
transmission=st.selectbox('Transmission',df['Transmission'].unique())
condition=st.selectbox('Condition',df['Condition'].unique())
aircon=st.selectbox('Air condition',df['Air condition'].unique())
mirror=st.selectbox('Power Mirrors',df['Power Mirrors'].unique())
key=st.selectbox('Keyless Entry',df['Keyless Entry'].unique())
steering=st.selectbox('Power Steering',df['Power Steering'].unique())
roof=st.selectbox('Sun Roof',df['Sun Roof'].unique())

if st.button("Predict price"):
    # Prepare input data
    user_input = np.array([company, model, year, km, fuel, city, document, assembly,
                           transmission, condition, aircon, mirror, key, steering, roof])
    user_input = user_input.reshape(1, -1)

    # Make Prediction
    predicted_price = int(np.exp(pipe.predict(user_input)[0]))

    # Display the Prediction with a meaningful message
    st.title(f'Your Car Price is: {predicted_price} PKR')







