# ==================== TRADE HUB ROUTES ====================

from trade_hub_templates import TRADE_HUB_HTML_TEMPLATE, LOGIN_HTML_TEMPLATE, REGISTER_HTML_TEMPLATE

# ==================== AUTHENTICATION ROUTES ====================
@app.route('/auth/register', methods=['GET', 'POST'])
def register():
    """User registration"""
    if request.method == 'GET':
        return render_template_string(REGISTER_HTML_TEMPLATE)
    
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        full_name = data.get('full_name', '').strip()
        phone = data.get('phone', '').strip()
        
        # Validation
        if not all([username, email, password, full_name]):
            return jsonify({'success': False, 'message': 'All fields are required'}), 400
        
        # Email validation
        try:
            validate_email(email)
        except EmailNotValidError:
            return jsonify({'success': False, 'message': 'Invalid email address'}), 400
        
        # Password strength validation
        is_strong, password_msg = validate_password_strength(password)
        if not is_strong:
            return jsonify({'success': False, 'message': password_msg}), 400
        
        db = get_db()
        
        # Check if user already exists
        existing_user = db.query(User).filter(
            (User.username == username) | (User.email == email)
        ).first()
        
        if existing_user:
            return jsonify({'success': False, 'message': 'Username or email already exists'}), 400
        
        # Create new user
        password_hash = generate_password_hash(password)
        new_user = User(
            username=username,
            email=email,
            password_hash=password_hash,
            full_name=full_name,
            phone=phone,
            is_verified=False,
            created_at=datetime.utcnow()
        )
        
        db.add(new_user)
        db.commit()
        
        # Create session
        session['user_id'] = new_user.id
        session['username'] = new_user.username
        
        return jsonify({
            'success': True,
            'message': 'Registration successful!',
            'user': {
                'id': new_user.id,
                'username': new_user.username,
                'email': new_user.email,
                'full_name': new_user.full_name
            }
        })
    
    except Exception as e:
        print(f"Registration error: {e}")
        return jsonify({'success': False, 'message': 'Registration failed. Please try again.'}), 500
    finally:
        db.close()

@app.route('/auth/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if request.method == 'GET':
        return render_template_string(LOGIN_HTML_TEMPLATE)
    
    try:
        data = request.get_json()
        username_or_email = data.get('username', '').strip()
        password = data.get('password', '')
        
        if not username_or_email or not password:
            return jsonify({'success': False, 'message': 'Username and password are required'}), 400
        
        db = get_db()
        
        # Find user by username or email
        user = db.query(User).filter(
            (User.username == username_or_email) | (User.email == username_or_email)
        ).first()
        
        if not user or not check_password_hash(user.password_hash, password):
            return jsonify({'success': False, 'message': 'Invalid credentials'}), 401
        
        # Update last login
        user.last_login = datetime.utcnow()
        db.commit()
        
        # Create session
        session['user_id'] = user.id
        session['username'] = user.username
        
        return jsonify({
            'success': True,
            'message': 'Login successful!',
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'full_name': user.full_name,
                'is_verified': user.is_verified,
                'is_premium': user.is_premium
            }
        })
    
    except Exception as e:
        print(f"Login error: {e}")
        return jsonify({'success': False, 'message': 'Login failed. Please try again.'}), 500
    finally:
        db.close()

@app.route('/auth/logout', methods=['POST'])
def logout():
    """User logout"""
    session.clear()
    return jsonify({'success': True, 'message': 'Logged out successfully'})

# ==================== LISTING ROUTES ====================
@app.route('/api/listings', methods=['GET'])
def get_listings():
    """Get listings with filters and pagination"""
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))
        category = request.args.get('category')
        location = request.args.get('location')
        min_price = request.args.get('min_price')
        max_price = request.args.get('max_price')
        search = request.args.get('search')
        sort_by = request.args.get('sort_by', 'created_at')
        sort_order = request.args.get('sort_order', 'desc')
        
        db = get_db()
        query = db.query(Listing).filter(Listing.is_active == True)
        
        # Apply filters
        if category:
            query = query.filter(Listing.category_id == category)
        
        if location:
            query = query.filter(Listing.location.ilike(f'%{location}%'))
        
        if min_price:
            query = query.filter(Listing.price >= float(min_price))
        
        if max_price:
            query = query.filter(Listing.price <= float(max_price))
        
        if search:
            search_term = f'%{search}%'
            query = query.filter(
                (Listing.title.ilike(search_term)) |
                (Listing.description.ilike(search_term)) |
                (Listing.tags.contains([search]))
            )
        
        # Apply sorting
        if sort_by == 'price':
            if sort_order == 'asc':
                query = query.order_by(Listing.price.asc())
            else:
                query = query.order_by(Listing.price.desc())
        elif sort_by == 'views':
            query = query.order_by(Listing.views.desc())
        elif sort_by == 'likes':
            query = query.order_by(Listing.likes.desc())
        else:  # created_at
            if sort_order == 'asc':
                query = query.order_by(Listing.created_at.asc())
            else:
                query = query.order_by(Listing.created_at.desc())
        
        # Pagination
        total = query.count()
        listings = query.offset((page - 1) * per_page).limit(per_page).all()
        
        # Convert to dict
        listings_data = []
        for listing in listings:
            listings_data.append({
                'id': listing.id,
                'title': listing.title,
                'description': listing.description,
                'price': listing.price,
                'currency': listing.currency,
                'location': listing.location,
                'state': listing.state,
                'district': listing.district,
                'condition': listing.condition,
                'images': listing.images or [],
                'tags': listing.tags or [],
                'is_negotiable': listing.is_negotiable,
                'views': listing.views,
                'likes': listing.likes,
                'created_at': listing.created_at.isoformat(),
                'ai_score': listing.ai_score,
                'trending_score': listing.trending_score,
                'user': {
                    'id': listing.user.id,
                    'username': listing.user.username,
                    'full_name': listing.user.full_name,
                    'rating': listing.user.rating,
                    'is_verified': listing.user.is_verified
                },
                'category': {
                    'id': listing.category.id,
                    'name': listing.category.name,
                    'icon': listing.category.icon
                }
            })
        
        return jsonify({
            'success': True,
            'listings': listings_data,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': total,
                'pages': math.ceil(total / per_page)
            }
        })
    
    except Exception as e:
        print(f"Error getting listings: {e}")
        return jsonify({'success': False, 'message': 'Failed to fetch listings'}), 500
    finally:
        db.close()

@app.route('/api/listings', methods=['POST'])
@login_required
def create_listing():
    """Create new listing"""
    try:
        data = request.get_json()
        
        # Validation
        required_fields = ['title', 'description', 'price', 'category_id', 'location', 'state', 'district']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'success': False, 'message': f'{field} is required'}), 400
        
        db = get_db()
        
        # Create listing
        new_listing = Listing(
            title=data['title'],
            description=data['description'],
            price=float(data['price']),
            currency=data.get('currency', 'INR'),
            category_id=int(data['category_id']),
            user_id=session['user_id'],
            location=data['location'],
            state=data['state'],
            district=data['district'],
            condition=data.get('condition', 'used'),
            images=data.get('images', []),
            tags=data.get('tags', []),
            is_negotiable=data.get('is_negotiable', True),
            created_at=datetime.utcnow()
        )
        
        # Calculate AI score
        new_listing.ai_score = calculate_ai_score(new_listing)
        
        db.add(new_listing)
        db.commit()
        
        return jsonify({
            'success': True,
            'message': 'Listing created successfully!',
            'listing_id': new_listing.id
        })
    
    except Exception as e:
        print(f"Error creating listing: {e}")
        return jsonify({'success': False, 'message': 'Failed to create listing'}), 500
    finally:
        db.close()

@app.route('/api/listings/<int:listing_id>')
def get_listing(listing_id):
    """Get single listing details"""
    try:
        db = get_db()
        listing = db.query(Listing).filter(
            Listing.id == listing_id,
            Listing.is_active == True
        ).first()
        
        if not listing:
            return jsonify({'success': False, 'message': 'Listing not found'}), 404
        
        # Increment view count
        listing.views += 1
        db.commit()
        
        # Get AI recommendations
        recommendations = []
        if TradeHubConfig.AI_ENABLED:
            all_listings = db.query(Listing).filter(
                Listing.is_active == True,
                Listing.id != listing_id
            ).limit(100).all()
            
            listings_data = []
            for l in all_listings:
                listings_data.append({
                    'id': l.id,
                    'title': l.title,
                    'description': l.description,
                    'tags': l.tags or []
                })
            
            recommendation_engine.train_model(listings_data)
            recommendations = recommendation_engine.get_recommendations(listing_id, listings_data)
        
        listing_data = {
            'id': listing.id,
            'title': listing.title,
            'description': listing.description,
            'price': listing.price,
            'currency': listing.currency,
            'location': listing.location,
            'state': listing.state,
            'district': listing.district,
            'condition': listing.condition,
            'images': listing.images or [],
            'tags': listing.tags or [],
            'is_negotiable': listing.is_negotiable,
            'views': listing.views,
            'likes': listing.likes,
            'created_at': listing.created_at.isoformat(),
            'ai_score': listing.ai_score,
            'trending_score': listing.trending_score,
            'user': {
                'id': listing.user.id,
                'username': listing.user.username,
                'full_name': listing.user.full_name,
                'rating': listing.user.rating,
                'total_ratings': listing.user.total_ratings,
                'is_verified': listing.user.is_verified,
                'location': listing.user.location,
                'bio': listing.user.bio
            },
            'category': {
                'id': listing.category.id,
                'name': listing.category.name,
                'icon': listing.category.icon
            },
            'recommendations': recommendations
        }
        
        return jsonify({
            'success': True,
            'listing': listing_data
        })
    
    except Exception as e:
        print(f"Error getting listing: {e}")
        return jsonify({'success': False, 'message': 'Failed to fetch listing'}), 500
    finally:
        db.close()

# ==================== MESSAGING ROUTES ====================
@app.route('/api/messages', methods=['GET'])
@login_required
def get_messages():
    """Get user messages"""
    try:
        db = get_db()
        user_id = session['user_id']
        
        # Get conversations
        conversations = db.query(Message).filter(
            (Message.sender_id == user_id) | (Message.receiver_id == user_id)
        ).order_by(Message.created_at.desc()).all()
        
        # Group by conversation
        conversation_map = {}
        for msg in conversations:
            other_user_id = msg.sender_id if msg.receiver_id == user_id else msg.receiver_id
            if other_user_id not in conversation_map:
                conversation_map[other_user_id] = {
                    'user_id': other_user_id,
                    'last_message': msg,
                    'unread_count': 0
                }
            
            if msg.receiver_id == user_id and not msg.is_read:
                conversation_map[other_user_id]['unread_count'] += 1
        
        conversations_data = []
        for conv in conversation_map.values():
            other_user = db.query(User).filter(User.id == conv['user_id']).first()
            conversations_data.append({
                'user': {
                    'id': other_user.id,
                    'username': other_user.username,
                    'full_name': other_user.full_name,
                    'profile_photo': other_user.profile_photo
                },
                'last_message': {
                    'content': conv['last_message'].content,
                    'created_at': conv['last_message'].created_at.isoformat(),
                    'is_read': conv['last_message'].is_read
                },
                'unread_count': conv['unread_count']
            })
        
        return jsonify({
            'success': True,
            'conversations': conversations_data
        })
    
    except Exception as e:
        print(f"Error getting messages: {e}")
        return jsonify({'success': False, 'message': 'Failed to fetch messages'}), 500
    finally:
        db.close()

@app.route('/api/messages', methods=['POST'])
@login_required
def send_message():
    """Send message"""
    try:
        data = request.get_json()
        receiver_id = data.get('receiver_id')
        content = data.get('content', '').strip()
        listing_id = data.get('listing_id')
        message_type = data.get('message_type', 'text')
        offer_price = data.get('offer_price')
        
        if not receiver_id or not content:
            return jsonify({'success': False, 'message': 'Receiver and content are required'}), 400
        
        db = get_db()
        
        # Create message
        new_message = Message(
            sender_id=session['user_id'],
            receiver_id=receiver_id,
            listing_id=listing_id,
            content=content,
            message_type=message_type,
            offer_price=offer_price,
            created_at=datetime.utcnow()
        )
        
        db.add(new_message)
        db.commit()
        
        # Create notification
        notification = Notification(
            user_id=receiver_id,
            title="New Message",
            message=f"You received a new message from {session['username']}",
            notification_type="message",
            data={'message_id': new_message.id, 'sender_id': session['user_id']},
            created_at=datetime.utcnow()
        )
        
        db.add(notification)
        db.commit()
        
        return jsonify({
            'success': True,
            'message': 'Message sent successfully!',
            'message_id': new_message.id
        })
    
    except Exception as e:
        print(f"Error sending message: {e}")
        return jsonify({'success': False, 'message': 'Failed to send message'}), 500
    finally:
        db.close()

# ==================== WATCHLIST ROUTES ====================
@app.route('/api/watchlist', methods=['GET'])
@login_required
def get_watchlist():
    """Get user watchlist"""
    try:
        db = get_db()
        user_id = session['user_id']
        
        watchlist_items = db.query(Watchlist).filter(
            Watchlist.user_id == user_id
        ).order_by(Watchlist.created_at.desc()).all()
        
        watchlist_data = []
        for item in watchlist_items:
            listing = item.listing
            watchlist_data.append({
                'id': item.id,
                'listing': {
                    'id': listing.id,
                    'title': listing.title,
                    'price': listing.price,
                    'currency': listing.currency,
                    'location': listing.location,
                    'images': listing.images or [],
                    'created_at': listing.created_at.isoformat()
                },
                'added_at': item.created_at.isoformat()
            })
        
        return jsonify({
            'success': True,
            'watchlist': watchlist_data
        })
    
    except Exception as e:
        print(f"Error getting watchlist: {e}")
        return jsonify({'success': False, 'message': 'Failed to fetch watchlist'}), 500
    finally:
        db.close()

@app.route('/api/watchlist', methods=['POST'])
@login_required
def add_to_watchlist():
    """Add listing to watchlist"""
    try:
        data = request.get_json()
        listing_id = data.get('listing_id')
        
        if not listing_id:
            return jsonify({'success': False, 'message': 'Listing ID is required'}), 400
        
        db = get_db()
        user_id = session['user_id']
        
        # Check if already in watchlist
        existing = db.query(Watchlist).filter(
            Watchlist.user_id == user_id,
            Watchlist.listing_id == listing_id
        ).first()
        
        if existing:
            return jsonify({'success': False, 'message': 'Already in watchlist'}), 400
        
        # Add to watchlist
        watchlist_item = Watchlist(
            user_id=user_id,
            listing_id=listing_id,
            created_at=datetime.utcnow()
        )
        
        db.add(watchlist_item)
        db.commit()
        
        return jsonify({
            'success': True,
            'message': 'Added to watchlist!'
        })
    
    except Exception as e:
        print(f"Error adding to watchlist: {e}")
        return jsonify({'success': False, 'message': 'Failed to add to watchlist'}), 500
    finally:
        db.close()

@app.route('/api/watchlist/<int:listing_id>', methods=['DELETE'])
@login_required
def remove_from_watchlist(listing_id):
    """Remove listing from watchlist"""
    try:
        db = get_db()
        user_id = session['user_id']
        
        watchlist_item = db.query(Watchlist).filter(
            Watchlist.user_id == user_id,
            Watchlist.listing_id == listing_id
        ).first()
        
        if not watchlist_item:
            return jsonify({'success': False, 'message': 'Not in watchlist'}), 404
        
        db.delete(watchlist_item)
        db.commit()
        
        return jsonify({
            'success': True,
            'message': 'Removed from watchlist!'
        })
    
    except Exception as e:
        print(f"Error removing from watchlist: {e}")
        return jsonify({'success': False, 'message': 'Failed to remove from watchlist'}), 500
    finally:
        db.close()

# ==================== ANALYTICS ROUTES ====================
@app.route('/api/analytics/dashboard')
@login_required
def get_analytics_dashboard():
    """Get user analytics dashboard"""
    try:
        db = get_db()
        user_id = session['user_id']
        
        # Get user's listings
        user_listings = db.query(Listing).filter(Listing.user_id == user_id).all()
        
        # Calculate stats
        total_listings = len(user_listings)
        active_listings = len([l for l in user_listings if l.is_active])
        total_views = sum(l.views for l in user_listings)
        total_likes = sum(l.likes for l in user_listings)
        
        # Recent activity
        recent_messages = db.query(Message).filter(
            (Message.sender_id == user_id) | (Message.receiver_id == user_id)
        ).order_by(Message.created_at.desc()).limit(10).all()
        
        # Price trends (simplified)
        price_trends = []
        for listing in user_listings[-10:]:  # Last 10 listings
            price_trends.append({
                'date': listing.created_at.isoformat(),
                'price': listing.price,
                'title': listing.title
            })
        
        analytics_data = {
            'overview': {
                'total_listings': total_listings,
                'active_listings': active_listings,
                'total_views': total_views,
                'total_likes': total_likes,
                'avg_views_per_listing': total_views / total_listings if total_listings > 0 else 0
            },
            'recent_activity': [
                {
                    'type': 'message',
                    'content': msg.content[:50] + '...' if len(msg.content) > 50 else msg.content,
                    'created_at': msg.created_at.isoformat()
                }
                for msg in recent_messages
            ],
            'price_trends': price_trends
        }
        
        return jsonify({
            'success': True,
            'analytics': analytics_data
        })
    
    except Exception as e:
        print(f"Error getting analytics: {e}")
        return jsonify({'success': False, 'message': 'Failed to fetch analytics'}), 500
    finally:
        db.close()

# ==================== SEARCH ROUTES ====================
@app.route('/api/search/advanced')
def advanced_search():
    """Advanced search with AI ranking"""
    try:
        query = request.args.get('q', '').strip()
        category = request.args.get('category')
        location = request.args.get('location')
        min_price = request.args.get('min_price')
        max_price = request.args.get('max_price')
        sort_by = request.args.get('sort_by', 'relevance')
        
        if not query:
            return jsonify({'success': False, 'message': 'Search query is required'}), 400
        
        db = get_db()
        
        # Build search query
        search_query = db.query(Listing).filter(
            Listing.is_active == True,
            (Listing.title.ilike(f'%{query}%')) |
            (Listing.description.ilike(f'%{query}%')) |
            (Listing.tags.contains([query]))
        )
        
        # Apply filters
        if category:
            search_query = search_query.filter(Listing.category_id == category)
        
        if location:
            search_query = search_query.filter(Listing.location.ilike(f'%{location}%'))
        
        if min_price:
            search_query = search_query.filter(Listing.price >= float(min_price))
        
        if max_price:
            search_query = search_query.filter(Listing.price <= float(max_price))
        
        # Get results
        results = search_query.limit(50).all()
        
        # AI-powered ranking
        if TradeHubConfig.AI_ENABLED and results:
            # Calculate relevance scores
            for listing in results:
                relevance_score = 0
                
                # Title match (highest weight)
                if query.lower() in listing.title.lower():
                    relevance_score += 10
                
                # Description match
                if query.lower() in listing.description.lower():
                    relevance_score += 5
                
                # Tags match
                if listing.tags and any(query.lower() in tag.lower() for tag in listing.tags):
                    relevance_score += 8
                
                # Combine with AI score and trending score
                listing.relevance_score = relevance_score + (listing.ai_score * 0.1) + (listing.trending_score * 0.05)
            
            # Sort by relevance
            results.sort(key=lambda x: getattr(x, 'relevance_score', 0), reverse=True)
        
        # Convert to response format
        search_results = []
        for listing in results:
            search_results.append({
                'id': listing.id,
                'title': listing.title,
                'description': listing.description,
                'price': listing.price,
                'currency': listing.currency,
                'location': listing.location,
                'images': listing.images or [],
                'ai_score': listing.ai_score,
                'relevance_score': getattr(listing, 'relevance_score', 0),
                'created_at': listing.created_at.isoformat(),
                'user': {
                    'username': listing.user.username,
                    'rating': listing.user.rating,
                    'is_verified': listing.user.is_verified
                }
            })
        
        return jsonify({
            'success': True,
            'query': query,
            'results': search_results,
            'total': len(search_results)
        })
    
    except Exception as e:
        print(f"Error in advanced search: {e}")
        return jsonify({'success': False, 'message': 'Search failed'}), 500
    finally:
        db.close()

# ==================== ERROR HANDLERS ====================
@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'message': 'Page not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    print(f"Internal server error: {error}")
    return jsonify({'success': False, 'message': 'Internal server error'}), 500