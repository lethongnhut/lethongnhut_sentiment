# References:
# [Streamlit API](https://docs.streamlit.io/library/api-reference)

# ====================== Import libraries ====================== #

import os
import streamlit as st
import pandas as pd
import utils


# ====================== Definitions and functions ====================== #
# --- Data path ---
ImgPath = './Misc/'
BrandImg                  = os.path.join(ImgPath, "Shopee_Logo.png")
CollaUserItemImg          = os.path.join(ImgPath, "CollaborativeFiltering_UserBased_ItemBased.png")
ContentBasedImg           = os.path.join(ImgPath, "content_based_filtering.jpg")
No_Image_Available        = os.path.join(ImgPath, "No_Image_Available.jpg")
# --- For GUI ---
BussinessObjective        = "Business Objective"
ContentBasedFiltering     = "Content-based Filtering"
CollaborativeFiltering    = "Collaborative Filtering"
menu_                     = [BussinessObjective, ContentBasedFiltering, CollaborativeFiltering]
FilterProdDesc            = "Product description"
FilterProdLst             = "Product list"
UserBasedFilter           = "User-based Filtering"
ItemBasedFilter           = "Item-based Filtering"
DEF_SIMILARITY_THRESHOLD  = 0.4
DEF_RATING_THRESHOLD      = 3.0


# ====================== Declarations ====================== #

pr_ = utils.ProductRecommendations()


# ====================== Streamlit GUI & Process ====================== #
def product_info_display(row):
  """Display product info in a grid
  Parameters
  ----------
  row : pandas.core.series.Series
      A row of the dataframe

  Returns
  -------
    None
  """ 
  # --- display product image ---
  if row.image != None:
    # Check if image can be displayed
    try:
      st.image(row.image, width=200, caption=f'ID {int(row.product_id)}', use_column_width='auto')
    except:
      st.image(No_Image_Available, width=200, caption=f'ID {int(row.product_id)}', use_column_width='auto')
  else:
    st.write('No image')
  # --- display product_id and product_name ---
  if row.link:
    # display product_name with maximum lines of 3
    st.markdown('[{}]({})'.format(row.product_name, row.link))
  else:
    st.write(row.product_name)

  return


def handle_cb_search_button_click(desc, rec_nums, threshold, isVoice):
  """Handle search button click
  Parameters
  ----------
  desc : str
      Product description
  rec_nums : int
      Number of recommended products
  isVoice : bool
      Whether the input is from voice or not

  Returns
  -------
    None
  """
  if isVoice:
    description = utils.takecomand()
  else:
    description = desc

  # Get top similar products
  results = pr_.recommend_products(description, rec_nums, threshold)
  # Check if the results is empty
  if results.empty:
    st.error('No similar products found!')
    return
  # Add separator
  st.markdown('---')
  # Display top similar products found
  st.write('Top {} similar products with similarity >= {}:'.format(results.shape[0], round(threshold, 2)))
  # Get info of the product
  results = pr_.get_product_info(results, 'product_id')
  st.write(results[['product_id', 'similarity', 'product_name', 'product_name_description_processed', 'image']])

  # create a grid with four columns and display the product images in 2 columns with the same size
  col1, col2, col3 = st.columns(3)
  for group_num, group in results.groupby((results.index % 3)):
    with col1 if group_num == 0 else col2 if group_num == 1 else col3:
      for row in group.itertuples():
          product_info_display(row)
  return


def handle_cf_user_search_button_click(user_id, rec_nums, threshold):
  """Handle collaborative user-based search button click
  Parameters
  ----------
  user_id : str
      User ID
  rec_nums : int
      Number of recommended products
  threshold : float
      Rating threshold
      
  Returns
  -------
    None
  """
  # --- Get top rating history of that user ---
  df_rating = pr_.get_top_user_rated_items(user_id)
  with st.expander('See rating history of user'):
    df_rating['link'] = df_rating['link'].apply(utils.make_clickable)
    df_rating = df_rating.to_html(escape=False)
    st.write(df_rating, unsafe_allow_html=True)

  # --- Get top recommended products ---
  results = pr_.get_rec_user_items(user_id, rec_nums, threshold)
  # Check if the results is empty
  if results.empty:
    st.error('No recommended products found!')
    return
  # Add separator
  st.markdown('---')
  # Display top recommended products found
  st.write('Top {} recommended products with rating >= {}:'.format(results.shape[0], round(threshold, 2)))
  # Get info of the product
  results = pr_.get_product_info(results, 'product_id')
  st.write(results[['product_id', 'product_name', 'rating', 'image', 'link']])
  # create a grid with four columns and display the product images in 2 columns with the same size
  col1, col2, col3 = st.columns(3)
  for group_num, group in results.groupby((results.index % 3)):
    with col1 if group_num == 0 else col2 if group_num == 1 else col3:
      for row in group.itertuples():
          product_info_display(row)
  return


def handle_cf_item_search_button_click(product_id, rec_nums, threshold):
  """Handle collaborative item-based search button click
  Parameters
  ----------
  product_id : str
      Product ID
  rec_nums : int
      Number of recommended products
  threshold : float
      Rating threshold
      
  Returns
  -------
    None
  """
  # Get top potential users
  results, df_rating = pr_.get_rec_item_users(product_id, rec_nums, threshold)
  # Check if the results is empty
  if results is None:
    st.error('No potential users found!')
    return
  # Get item info
  item_info_  = pr_.get_product_info_(product_id)
  product_info_display(item_info_.iloc[0])

  # Add separator
  st.markdown('---')
  # Display top potential users found
  st.write('Top {} potential customers with rating >= {}:'.format(results.shape[0], round(threshold, 2)))
  st.write(results[['user_id', 'user', 'rating']])
  st.markdown('---')
  # Display rating history of users
  with st.expander('See rating history of users'):
    df_rating['link'] = df_rating['link'].apply(utils.make_clickable)
    df_rating = df_rating.to_html(escape=False)
    st.write(df_rating, unsafe_allow_html=True)
  return


def content_gui(desc, isVoice=False):
  """Content-based filtering GUI
  Parameters
  ----------
  desc : str
      Product description
  isVoice : bool
      Whether the input is from voice or not

  Returns
  -------
    None
  """
  if desc is None:
    if isVoice:
      description = None
    else:
      # Add text input to input product's ID or description
      description = st.text_input('Product ID or Description',
                                  help='Input product ID or description here.',
                                  )
  else:
    description = desc

  # Add slider and set default number of recommendations to 5
  # Number input threshold selecting
  col1, col2 = st.columns(2)
  # Use col1 and col2 to display the slider
  rec_nums = col1.slider('Number of Recommendations',
                          min_value=1,
                          max_value=10,
                          value=5,
                          step=1)
  threshold = col2.number_input('Similarity Threshold',
                              min_value=0.0,
                              max_value=1.0,
                              value=DEF_SIMILARITY_THRESHOLD,
                              step=0.05,
                              help='Similarity threshold (0.0 ~ 1.0) to filter products')
  # Add button to search
  search_button = st.form_submit_button(label='Search')
  if search_button:
    handle_cb_search_button_click(description, rec_nums, threshold, isVoice)
  return


def user_gui(user_id):
  """User-based filtering GUI
  Parameters
  ----------
  user_id : str
      User ID

  Returns
  -------
    None
  """
  # Add slider and set default number of recommendations to 5
  # Number input threshold selecting
  col1, col2 = st.columns(2)
  # Use col1 and col2 to display the slider
  rec_nums = col1.slider('Number of Recommendations',
                          min_value=1,
                          max_value=5,
                          value=5,
                          step=1)
  threshold = col2.number_input('Rating Threshold',
                              min_value=0.0,
                              max_value=5.0,
                              value=DEF_RATING_THRESHOLD,
                              step=0.05,
                              help='Rating threshold (0.0 ~ 5.0) to filter products')
  # Add button to search
  search_button = st.form_submit_button(label='Search')
  if search_button:
    handle_cf_user_search_button_click(user_id, rec_nums, threshold)
  return


def item_gui(item_id):
  """Item-based filtering GUI
  Parameters
  ----------
  product_id : str
      Product ID

  Returns
  -------
    None
  """
  # Add slider and set default number of recommendations to 5
  # Number input threshold selecting
  col1, col2 = st.columns(2)
  # Use col1 and col2 to display the slider
  rec_nums = col1.slider('Number of Recommendations',
                          min_value=1,
                          max_value=5,
                          value=5,
                          step=1)
  threshold = col2.number_input('Rating Threshold',
                              min_value=0.0,
                              max_value=5.0,
                              value=DEF_RATING_THRESHOLD,
                              step=0.05,
                              help='Rating threshold (0.0 ~ 5.0) to filter products')
  # Add button to search
  search_button = st.form_submit_button(label='Search')
  if search_button:
    handle_cf_item_search_button_click(item_id, rec_nums, threshold)
  return


def content_based_filtering(filter_option, isVoice=False):
  """Content-based filtering
  Parameters
  ----------
  filter_option : str
      Option to filter products

  Returns
  -------
    None
  """
  if filter_option == FilterProdDesc:
    # Stick widgets
    with st.form(key='my_form'):
      content_gui(None, isVoice)
  elif filter_option == FilterProdLst:
    # Stick widgets
    with st.form(key='my_form'):
      product_info = st.selectbox("Select a product", pr_.get_product_id_name_list())
      # Extract product_id from product_info
      product_id = product_info.split(' - ')[0]
      content_gui(product_id)
  return


def collaborative_based_filtering(filter_option):
  """Collaborative-based filtering
  Parameters
  ----------
  filter_option : str
      Option to filter products

  Returns
  -------
    None
  """
  if filter_option == UserBasedFilter:
    # --- Get top user with rating ---
    df_rating = pr_.get_top_user_with_rating()
    with st.expander('See top user with ratings'):
      st.write(df_rating)

    # Stick widgets
    with st.form(key='my_form'):
      # Select user's ID
      user_id_name = st.selectbox("Select User ID", pr_.get_all_user_ids_names())
      user_id = int(user_id_name.split(' - ')[0])
      user_gui(user_id)
  elif filter_option == ItemBasedFilter:
    # Stick widgets
    with st.form(key='my_form'):
      item_id_name  = st.selectbox("Select Item ID",
                              pr_.get_all_item_ids_names(),
                              # format_func=lambda x: x.split(' - ')[0],
                              )
      item_id = int(item_id_name.split(' - ')[0])
      item_gui(item_id)
  return


def main():
  # --- Sidebar --- 
  # Add title to the sidebar
  st.sidebar.title('Shopee Recommendation System')
  # Use radio button to choose between content-based filltering and collaborative filtering
  page = st.sidebar.radio('Menu', menu_)
  if page == BussinessObjective:
    st.title('Data Science Capstone Project')
    st.subheader("Shopee Recommendation System")
    st.image(BrandImg, width=400)
    st.subheader("***Business Understanding***")
    # --- Content-based Filtering ---
    st.write("## Content-based Filtering")
    st.write("Content-based filtering uses item features to recommend other items similar to what the user likes, based on their previous actions or explicit feedback.")
    st.image(ContentBasedImg, width=800)
    # --- Collaborative Filtering ---
    st.write("## Collaborative Filtering")
    st.write("Collaborative filtering is a technique that can filter out items that a user might like on the basis of reactions by similar users.\
              It works by searching a large group of people and finding a smaller set of users with tastes similar to a particular user.\
              The system would then recommend items that those similar users liked to the particular user.")
    st.image(CollaUserItemImg, width=800)
    
  # --- Content-based Filtering ---
  elif page == ContentBasedFiltering:
    desc_ = None
    st.title('Data Science Capstone Project')
    st.subheader("Shopee Recommendation System")
    st.image(ContentBasedImg, width=700)
    st.write("# Content-based Filtering")
    # Add select box to choose between product list and manual input
    option = st.sidebar.selectbox("Select option for fitler", [FilterProdDesc, FilterProdLst])
    if option == FilterProdDesc:
      isVoice = st.sidebar.checkbox('By voice (Vnmese) 🎙️')
    # Add brief description on the sidebar
    st.sidebar.markdown('---')
    st.sidebar.title('About')
    st.sidebar.info('This is a search engine for fashion products. You can search for products by inputting product ID or description.\
                     The search engine will return the most similar products based on the input.')
    # Add separator
    st.sidebar.markdown('---')
    # A brief description about the search engine
    st.sidebar.title('Techniques')
    st.sidebar.write("The search engine uses TF-IDF to vectorize the product descriptions then **Gemsim** and **Cosine similarity** are used to calculate the similarity between products.")
    st.sidebar.title('Conclusion')
    st.sidebar.markdown("- Gemsim is prefer to use because it is more flexible and can be used with input is text description or product's ID.")
    st.sidebar.markdown('- Saving and storing model is also easier with Gemsim.')
    if option == FilterProdDesc:
      content_based_filtering(option, isVoice)
    elif option == FilterProdLst:
      content_based_filtering(option)

  # --- CollaborativeFiltering ---
  elif page == CollaborativeFiltering:
    st.title('Data Science Capstone Project')
    st.subheader("Shopee Recommendation System")
    st.image(CollaUserItemImg, width=700)
    st.write("# Collaborative Filtering")
    # Add select box to choose between product list and manual input
    option = st.sidebar.selectbox("Select option for fitler", [UserBasedFilter, ItemBasedFilter])
    # Add brief description on the sidebar
    st.sidebar.markdown('---')
    st.sidebar.title('About')
    st.sidebar.info("This is a search engine for fashion products")
    st.sidebar.info("- User-based filtering: The search engine will return the most recommended products based on the user's ID.")
    st.sidebar.info("- Item-based filtering: The search engine will return potential customers for that product.")
    # Add separator
    st.sidebar.markdown('---')
    # A brief description about the search engine
    st.sidebar.title('Techniques')
    st.sidebar.write("Algorithm: **Alternating Least Squares (ALS)**")
    st.sidebar.subheader('Cross-Validator')
    rank_       = [10, 40]
    max_iter_   = [10]
    reg_param_  = [0.01, 0.1]
    alpha_      = [1.0]
    numFolds_   = 3
    ParamMaps = { 'Parameter': ['rank', 'maxIter', 'regParam', 'alpha', 'numFolds'],
                  'Value': [rank_, max_iter_, reg_param_, alpha_, numFolds_]
                }
    st.sidebar.write(pd.DataFrame(ParamMaps))
    st.sidebar.subheader('Hyperparameters (RMSE = 1.18)')
    df_hyper = pd.DataFrame({'rank': ['40',],
                            'maxIter': ['10',],
                            'regParam': ['0.1',],
                            'alpha': ['1.0',],})
    st.sidebar.table(df_hyper)
    st.sidebar.markdown(':blue[***Note:***] The hyperparameters are chosen by using cross-validation with 3 folds.\
                        Increasing value of **rank** will increase the accuracy of the model but will also increase the training time and memory usage.\
                        Pay attention to this parameter to not make the model overfitting.')
    collaborative_based_filtering(option)
# ====================== Main ====================== #
if __name__ == "__main__":
  main()