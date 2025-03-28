// Rust style structs and functions
pub struct User {
    id: String,
    username: String,
    active: bool,
}

impl User {
    pub fn new(username: String) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            username,
            active: true,
        }
    }

    fn is_active(&self) -> bool {
        self.active
    }
}

pub struct UserService {
    users: std::collections::HashMap<String, User>,
}

impl UserService {
    pub fn new() -> Self {
        Self {
            users: std::collections::HashMap::new(),
        }
    }

    pub async fn create_user(&mut self, username: String) -> Result<User, String> {
        let user = User::new(username);
        self.users.insert(user.id.clone(), user.clone());
        Ok(user)
    }

    fn validate_user(&self, user: &User) -> bool {
        user.is_active()
    }
}

// Free functions
pub fn process_user(user: &User) -> bool {
    user.is_active()
}

async fn fetch_user_data(id: &str) -> Result<User, String> {
    // Simulated async operation
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    Err("Not implemented".to_string())
} 