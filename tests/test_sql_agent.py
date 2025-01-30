import logging

import sqlite3


from openvela.agents import Agent, SQLAgent, SupervisorAgent
from openvela.llms import GroqModel, OllamaModel
from openvela.tasks import Task
from openvela.workflows import AutoSelectWorkflow


# ------------------------------------------------------------------
# 1. Configure Logging
# ------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)

# ------------------------------------------------------------------
# 2. Initialize the LLM (GroqModel)
#    Replace api_key with your actual key.
# ------------------------------------------------------------------
model_instance = OllamaModel(
    model="mistral",
)
# If using Ollama locally, you could do:
# model_instance = OllamaModel(model="llama3.1")
model_supervisor = OllamaModel(model="mistral")


# ------------------------------------------------------------------
# 3. Setup a Complex SQLite Database: 8 Tables + Sample Data
# ------------------------------------------------------------------
def setup_database():
    """
    Creates or resets 'example.db' with 8 tables that demonstrate
    relationships, foreign keys, and sample data.
    """
    conn = sqlite3.connect("example.db")
    cursor = conn.cursor()

    # Enable foreign keys
    cursor.execute("PRAGMA foreign_keys = ON;")

    cursor.executescript(
        """
        ----------------------------------------------------------------------------
        -- Drop existing tables if they exist
        ----------------------------------------------------------------------------
        DROP TABLE IF EXISTS payments;
        DROP TABLE IF EXISTS product_reviews;
        DROP TABLE IF EXISTS order_items;
        DROP TABLE IF EXISTS orders;
        DROP TABLE IF EXISTS products;
        DROP TABLE IF EXISTS categories;
        DROP TABLE IF EXISTS addresses;
        DROP TABLE IF EXISTS users;

        ----------------------------------------------------------------------------
        -- Create tables
        ----------------------------------------------------------------------------

        -- 1) users
        CREATE TABLE users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        -- 2) addresses (references users)
        CREATE TABLE addresses (
            address_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            address_line1 TEXT NOT NULL,
            address_line2 TEXT,
            city TEXT NOT NULL,
            state TEXT NOT NULL,
            zip_code TEXT NOT NULL,
            country TEXT NOT NULL,
            is_primary BOOLEAN NOT NULL DEFAULT 0,
            FOREIGN KEY(user_id) REFERENCES users(user_id)
        );

        -- 3) categories
        CREATE TABLE categories (
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT NOT NULL,
            parent_category_id INTEGER,
            FOREIGN KEY(parent_category_id) REFERENCES categories(category_id)
        );

        -- 4) products (references categories)
        CREATE TABLE products (
            product_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_id INTEGER NOT NULL,
            product_name TEXT NOT NULL,
            description TEXT,
            price NUMERIC NOT NULL,
            stock INTEGER NOT NULL DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(category_id) REFERENCES categories(category_id)
        );

        -- 5) orders (references users)
        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            order_date DATETIME DEFAULT CURRENT_TIMESTAMP,
            status TEXT NOT NULL DEFAULT 'Pending',
            FOREIGN KEY(user_id) REFERENCES users(user_id)
        );

        -- 6) order_items (references orders, products)
        CREATE TABLE order_items (
            order_item_id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id INTEGER NOT NULL,
            product_id INTEGER NOT NULL,
            quantity INTEGER NOT NULL,
            price NUMERIC NOT NULL, -- snapshot price at order time
            FOREIGN KEY(order_id) REFERENCES orders(order_id),
            FOREIGN KEY(product_id) REFERENCES products(product_id)
        );

        -- 7) product_reviews (references products, users)
        CREATE TABLE product_reviews (
            review_id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            rating INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
            review_text TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(product_id) REFERENCES products(product_id),
            FOREIGN KEY(user_id) REFERENCES users(user_id)
        );

        -- 8) payments (references orders)
        CREATE TABLE payments (
            payment_id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id INTEGER NOT NULL,
            amount NUMERIC NOT NULL,
            payment_method TEXT NOT NULL,
            payment_date DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(order_id) REFERENCES orders(order_id)
        );

        ----------------------------------------------------------------------------
        -- Insert sample data
        ----------------------------------------------------------------------------
        -- users
        INSERT INTO users (username, email) VALUES
            ('alice', 'alice@example.com'),
            ('bob', 'bob@example.com'),
            ('charlie', 'charlie@example.com'),
            ('david', 'david@example.com'),
            ('eve', 'eve@example.com');

        -- addresses
        INSERT INTO addresses (user_id, address_line1, city, state, zip_code, country, is_primary) VALUES
            (1, '123 Apple St.', 'Springfield', 'IL', '62701', 'USA', 1),
            (1, 'PO Box 45', 'Springfield', 'IL', '62702', 'USA', 0),
            (2, '10 Baker Ave.', 'Boston', 'MA', '02108', 'USA', 1),
            (3, '50 Cherry Rd.', 'Columbus', 'OH', '43210', 'USA', 1),
            (4, '77 Delta Dr.', 'Austin', 'TX', '73301', 'USA', 1),
            (5, '101 Elm St.', 'Seattle', 'WA', '98101', 'USA', 1);

        -- categories
        INSERT INTO categories (category_name) VALUES
            ('Electronics'),
            ('Books'),
            ('Clothing'),
            ('Kitchen');

        -- subcategories
        INSERT INTO categories (category_name, parent_category_id) VALUES
            ('Mobile Phones', 1),
            ('Laptops', 1),
            ('Fiction', 2),
            ('Non-fiction', 2);

        -- products
        INSERT INTO products (category_id, product_name, description, price, stock) VALUES
            (1, 'Smart TV', '4K Ultra HD', 399.99, 10),
            (2, 'Sci-Fi Novel', 'A popular science fiction book.', 14.99, 30),
            (3, 'T-Shirt', 'Cotton T-Shirt for men', 9.99, 50),
            (4, 'Chef Knife', '8-inch professional chef knife', 39.99, 20),
            (5, 'Android Phone', 'Latest Android smartphone', 699.99, 15),
            (6, 'Gaming Laptop', 'High performance gaming laptop', 1299.99, 8),
            (7, 'Mystery Novel', 'Suspense thriller novel', 12.99, 25),
            (8, 'History Book', 'A deep dive into historical events', 19.99, 18);

        -- orders
        INSERT INTO orders (user_id, status) VALUES
            (1, 'Pending'),
            (1, 'Shipped'),
            (2, 'Completed'),
            (3, 'Pending'),
            (4, 'Pending'),
            (5, 'Completed');

        -- order_items
        INSERT INTO order_items (order_id, product_id, quantity, price) VALUES
            (1, 1, 1, 399.99),
            (1, 3, 2, 9.99),
            (2, 5, 1, 699.99),
            (3, 2, 1, 14.99),
            (3, 7, 1, 12.99),
            (4, 6, 1, 1299.99),
            (5, 4, 1, 39.99),
            (6, 8, 2, 19.99);

        -- product_reviews
        INSERT INTO product_reviews (product_id, user_id, rating, review_text) VALUES
            (1, 1, 5, 'Excellent picture quality.'),
            (5, 1, 4, 'Great phone, but battery life is average.'),
            (6, 3, 4, 'Awesome performance.'),
            (2, 2, 5, 'A must-read for sci-fi fans.'),
            (8, 5, 3, 'Interesting read, but a bit dry.');

        -- payments
        INSERT INTO payments (order_id, amount, payment_method) VALUES
            (3, 29.98, 'Credit Card'),
            (5, 39.99, 'PayPal'),
            (6, 39.98, 'Debit Card');
        """
    )

    conn.commit()
    conn.close()


# Run the setup
setup_database()

# ------------------------------------------------------------------
# 4. Provide a schema summary for context
# ------------------------------------------------------------------
schema_summary = """
Tables:
  1) users (user_id, username, email, created_at)
  2) addresses (address_id, user_id, address_line1, address_line2, city, state, zip_code, country, is_primary)
  3) categories (category_id, category_name, parent_category_id)
  4) products (product_id, category_id, product_name, description, price, stock, created_at)
  5) orders (order_id, user_id, order_date, status)
  6) order_items (order_item_id, order_id, product_id, quantity, price)
  7) product_reviews (review_id, product_id, user_id, rating, review_text, created_at)
  8) payments (payment_id, order_id, amount, payment_method, payment_date)

Relationships:
  - addresses.user_id => users.user_id
  - products.category_id => categories.category_id
  - orders.user_id => users.user_id
  - order_items.order_id => orders.order_id
  - order_items.product_id => products.product_id
  - product_reviews.product_id => products.product_id
  - product_reviews.user_id => users.user_id
  - payments.order_id => orders.order_id
"""

# ------------------------------------------------------------------
# 5. Define Four SQLAgents, each responsible for two related tables
# ------------------------------------------------------------------

# (A) Agent for users + addresses
users_addresses_agent = SQLAgent(
    settings={
        "name": "UsersAddressesAgent",
        "prompt": "Generate a read-only SQL query for the 'users' and 'addresses' tables.",
        "description": "Specialized Agent for users & addresses queries (read-only).",

    },
    model=model_instance,
    sql_dialect="sqlite",
    sqlalchemy_engine_url="sqlite:///example.db",
    database_structure=schema_summary,
)


# (B) Agent for categories + products
catalog_agent = SQLAgent(
    settings={
        "name": "CatalogAgent",
        "prompt": "Generate a read-only SQL query for the 'categories' and 'products' tables.",
        "description": "Specialized Agent for categories & products queries (read-only).",

    },
    model=model_instance,
    sql_dialect="sqlite",
    sqlalchemy_engine_url="sqlite:///example.db",
    database_structure=schema_summary,
)

# (C) Agent for orders + order_items
orders_agent = SQLAgent(
    settings={
        "name": "OrdersAgent",
        "prompt": "Generate a read-only SQL query for 'orders' and 'order_items' tables.",
        "description": "Specialized Agent for orders & order_items queries (read-only).",

    },
    model=model_instance,
    sql_dialect="sqlite",
    sqlalchemy_engine_url="sqlite:///example.db",
    database_structure=schema_summary,
)


# (D) Agent for product_reviews + payments
reviews_payments_agent = SQLAgent(
    settings={
        "name": "ReviewsPaymentsAgent",
        "prompt": "Generate a read-only SQL query for 'product_reviews' and 'payments' tables.",
        "description": "Specialized Agent for product_reviews & payments queries (read-only).",

    },
    model=model_instance,
    sql_dialect="sqlite",
    sqlalchemy_engine_url="sqlite:///example.db",
    database_structure=schema_summary,
)


# ------------------------------------------------------------------
# 6. SupervisorAgent to coordinate the Agents
# ------------------------------------------------------------------
supervisor = SupervisorAgent(
    settings={
        "name": "SupervisorAgent",
        "prompt": (
            "never create a json input"
            "before finish you may also select a formatting agent to produce a cohesive result."
        ),
        "description": "Supervises the workflow and manages the sequence of specialized Agents.",
    },
    model=model_supervisor,
    agent_type="selector",
)

# Optional: a FormatterAgent to finalize output
formatter_agent = Agent(
    settings={
        "name": "FormatterAgent",
        "prompt": "Format all partial SQL queries or responses into a cohesive final answer.",
        "description": "Formats the final text to produce a user-friendly output.",
    },
    model=model_instance,
)

# ------------------------------------------------------------------
# 7. Define a Complex Task that Touches All Tables
#    (We want the user request to require data from each pair of tables.)
# ------------------------------------------------------------------
complex_prompt = (
    "Give me a list of all users, each user's addresses, "
    "the categories of products they have ordered, the total quantity of items, "
    "the sum of payments made, and any review ratings they've given. "
    "Sort the result by user_id ascending."
)
task = Task(prompt=complex_prompt)

# ------------------------------------------------------------------
# 8. Build and Run the Workflow
# ------------------------------------------------------------------
workflow = AutoSelectWorkflow(
    task=task,
    agents=[
        users_addresses_agent,
        catalog_agent,
        orders_agent,
        reviews_payments_agent,
        formatter_agent,  # We can also pass the FormatterAgent if we want it considered
    ],

    supervisor=supervisor,
    validate_output=False,
    max_attempts=3,
)


final_output = workflow.run()

# ------------------------------------------------------------------
# 9. Print the Final Output
# ------------------------------------------------------------------

print("Final Output:\n", final_output)
