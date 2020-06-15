import pandas as pd
import streamlit as st

if __name__ == "__main__":

    recipe_file = "/Users/Carol/Dropbox/epicurious-recipes-with-rating-and-nutrition/full_format_recipes.json"
    recipe_df = pd.read_json(recipe_file, orient='records')
    recipe_df['categories'] = recipe_df['categories'].fillna(value="")
    # cats = list(set(recipe_df.categories.tolist()))
    # cats.sort()
    st.dataframe(recipe_df)


