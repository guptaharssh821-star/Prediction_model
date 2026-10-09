import streamlit as st
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from Train_test_split_ import One_encoder
import seaborn as sns
import matplotlib.pyplot as plt


class Mmultiple_Regression:
    def __init__ (self , Data_path , target_column):

        self.Data = Data_path
        self.target_column = target_column


# FUNTION 2 FOR CLEANING AND CALCULATION OF SLOPS

    def fit(self):

        Data = self.Data.copy()
        print(Data.columns)
        Data = Data.drop(['title','Brand me','price_per_sqft'],axis=1,errors='ignore')
        Data = Data.dropna()
        print(Data.shape)


        self.X = Data.drop(self.target_column,axis=1)
        self.Y = Data[self.target_column]


        # Separate the numerical and categorical columns 
        
        self.num_column = self.X.select_dtypes(include = np.number).columns
        self.cat_column = self.X.select_dtypes(exclude = np.number).columns

        # Encode categorical columns 

        # self.column = self.X.columns

        # print(self.X.head(30))

        # Train_test_split

        self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(
            self.X,
            self.Y.to_numpy(),
            test_size=0.20,
            random_state=42
            )

        # Create encoder

        self.encoder = OneHotEncoder(
            drop='first',
            handle_unknown='ignore',
            sparse_output=False
        )

        # Fit Encoder on Training Data 

        Train_Encoded = self.encoder.fit_transform(
            self.x_train[self.cat_column]
        )

        Test_Encoded = self.encoder.transform(               # Here we didnt use the the fit() Because it already--                                                   
            self.x_test[self.cat_column]                     # learned the patter of converting into 1,0 in privious use 
        )

        Encoded_Columns = self.encoder.get_feature_names_out()

        

        # Convert Encoded Arrays into DataFrame

        Train_Encoded = pd.DataFrame(
            Train_Encoded,
            columns=Encoded_Columns,
            index=self.x_train.index
        )

        Test_Encoded = pd.DataFrame(
            Test_Encoded,
            columns=Encoded_Columns,
            index=self.x_test.index
        )



        # select Numerical Columns 

        Train_Num = self.x_train[self.num_column]
        Test_Num = self.x_test[self.num_column]

        # combine Numerical and Categorical COlumns 0

        self.x_train = pd.concat([Train_Num , Train_Encoded],axis=1)
        self.x_test  = pd.concat([Test_Num  , Test_Encoded],axis=1)

        # convert it in Numpy Array 


        self.x_train = self.x_train.to_numpy()
        self.x_test  = self.x_test.to_numpy()



        # ADD Intercept

        X_train = np.column_stack((
            np.ones(len(self.x_train)),
            self.x_train
        )) 

        # Calculate Coefficients
        # Formula 
        # b^ = (self.X.T self.X)^-1 self.X.T y

        self.B = np.linalg.pinv(X_train) @ self.y_train

        self.intercept = self.B[0]
        self.Slop = self.B[1:]

        print("Intercept : ",self.intercept)
        print("Slopes    : ",self.Slop)

    def predict(self):
        X_test = np.column_stack((
            np.ones(len(self.x_test)),
            self.x_test
        ))

        self.Y_pred = X_test @ self.B

        self.residuals = self.y_test - self.Y_pred

        mse = np.mean(self.residuals**2)
        rmse = np.sqrt(mse)

        self.mse = mse
        self.rmse = rmse

        Compare = pd.DataFrame({
            'Actual': self.y_test[:20],
            'Predicted':self.Y_pred[:20],
            'Residuals': self.residuals[:20]
            

            })
        

        print("MSE  : ",mse)
        print("RMSE : ",rmse)
        print(Compare)
        
        negative = np.sum(self.Y_pred < 0)
        total = len(self.Y_pred)

        print("Negative Predictions :", negative)
        print("Total Predictions    :", total)
        print("Negative Percentage  :", (negative / total) * 100, "%")

        print("Numerical columns:")
        print(self.num_column)
        print("\nCategorical columns:")
        print(self.cat_column)
        print("\nTraining shape:")
        print(self.x_train.shape)
        print("\nTesting shape:")
        print(self.x_test.shape)

        print("\nTarget:")
        print(self.y_train[:20])



    def New_Prediction(self,New_Data):

        # New_Data = {}
        # print("\nEnter values for New Prediction\n")

        # # Numerical values
        # for col in self.num_column:
        #     value = float(input(f"Enter {col}: "))
        #     New_Data[col] = value

        # # Categorical values
        # for col in self.cat_column:
        #         value = input(f"Enter {col}: ")
        #         New_Data[col] = value

       
        # Convert the New Data Into Data Frame 
        New_Data = pd.DataFrame([New_Data])

        # Seperate the Numerical and Categorical Data 
        New_num = New_Data[self.num_column]
        New_cat = New_Data[self.cat_column]

        New_Encoded = self.encoder.transform(New_cat)
        Encoded_column = self.encoder.get_feature_names_out()

        # Convert encoded data into DataFrame

        New_Encoded = pd.DataFrame(
            New_Encoded,
            columns=Encoded_column
        )

        New_Data = pd.concat(
            [
            New_num.reset_index(drop=True),
            New_Encoded.reset_index(drop=True)
            ],
            axis=1
        )

        # convert it into Numpy 

        New_Data = New_Data.to_numpy()

        # Add Intercept

        New_Data = np.column_stack((
            np.ones(len(New_Data)),New_Data
        ))

        # Prediction 

        Prediction = New_Data @ self.B

        print("Predicted Price: ", Prediction[0])

        return Prediction[0]

    def Show_Graph(self):
         Pred_y = self.Y_pred
         Test_y = self.y_test
         sns.displot(Test_y - Pred_y,kind='kde')
         plt.show()

st.set_page_config(
    page_title="Multiple Linear Regression",
    page_icon="🏠",
    layout="centered"
)
st.title(" 🧠 Machine Learning Prediction Model",text_alignment='center')
st.write("A general-purpose Multiple Linear " \
"Regression model implemented from scratch.")
upload_file = st.file_uploader(
    "Upload your CSV File Here",
    type=["CSV"],
    )

if upload_file is not None:
    Data = pd.read_csv(upload_file)

    st.subheader("Data Set")
    st.dataframe(Data.head(10),width='stretch')

    tragetColumn = st.selectbox(
        "Target Columns",
        Data.columns
    )

    if st.button(
         "🚀Train Model",
         type='primary'
         ):
            models = Mmultiple_Regression(Data,tragetColumn)
            models.fit()
            models.predict()

            st.session_state.models = models
            st.session_state.trained = True

            st.success("Models Trained Successfully ! ")

    if st.session_state.get("trained", False):
         models = st.session_state.models

         st.subheader("📈 Models Results",)
         col1 , col2 = st.columns(2)


         with col1:
              st.metric(
                   "MSE",
                   f"{models.mse:,.2f}"
              )

         with col2:
              st.metric(
                   "RMSE",
                   f"{models.rmse:,.2f}"
              )


         st.subheader("🔍 Actual VS Pridicted")

         Compare = pd.DataFrame(
              {
                     'Actual': models.y_test[:20],
                     'Predicted':models.Y_pred[:20],
                     'Residuals': models.residuals[:20]
         
              }
            )
         st.dataframe(
              Compare,
              use_container_width=True
         )

         st.subheader("📊 Residual Distribution")
         fig = models.Show_Graph()
         st.pyplot(fig,)

    st.subheader(f"🏠 Predict {tragetColumn} for New Data")
    st.write(
         "#### Enter the Value of New Data below :-"
    )

    New_Data = {}

    for col in models.num_column:
         New_Data[col] = st.number_input(
              f"Enter : {col}",
              value=0.0
         )
    st.write(
         "#### Enter The Value Categorical Value :-"
    )
    for col in models.cat_column:

         index = list(
              models.cat_column
        ).index(col)
         
         categories = (
              models.encoder
              .categories_[index]
              .tolist()
        )
         New_Data[col] = st.selectbox(
              f"Select {col}",
              categories
        )

    if st.button(" Predict House Price"):
         
         prediction = models.New_Prediction(
              New_Data
         )


         st.success(
              f"Predicted {tragetColumn}:{prediction:,.2f}"
         )











 
