// User authentication logic (2FA, role-based access)

const crypto = require('crypto');
const readline = require('readline');
const { authenticator } = require('otplib');

class AuthManager {
  constructor() {
    // Initialize authentication variables
    this.users = [
      {
        username: 'doctor1',
        hashedPassword: this.hashPassword('password123'),
        role: 'doctor',
        totpSecret: 'JBSWY3DPEHPK3PXP', // Example TOTP secret
      },
      // Add more users as needed
    ];
  }

  hashPassword(password) {
    // Hash the password using SHA-256 (for simplicity)
    return crypto.createHash('sha256').update(password).digest('hex');
  }

  async authenticate() {
    // Prompt user for username and password
    const username = await this.prompt('Username: ');
    const password = await this.prompt('Password: ');

    const user = this.users.find(u => u.username === username);
    if (!user || user.hashedPassword !== this.hashPassword(password)) {
      console.error('Authentication failed: Invalid username or password.');
      return false;
    }

    // Verify TOTP
    const totpCode = await this.prompt('Enter your 2FA code: ');
    const isValidTotp = authenticator.check(totpCode, user.totpSecret);
    if (!isValidTotp) {
      console.error('Authentication failed: Invalid 2FA code.');
      return false;
    }

    console.log('Authentication successful.');
    this.currentUser = user;
    return true;
  }

  prompt(question) {
    // Implementation for prompting user input
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
    });
    return new Promise(resolve => {
      rl.question(question, answer => {
        rl.close();
        resolve(answer);
      });
    });
  }

  getCurrentUserRole() {
    return this.currentUser ? this.currentUser.role : null;
  }
}

module.exports = AuthManager;